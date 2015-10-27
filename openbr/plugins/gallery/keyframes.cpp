/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2012 The MITRE Corporation                                      *
 *                                                                           *
 * Licensed under the Apache License, Version 2.0 (the "License");           *
 * you may not use this file except in compliance with the License.          *
 * You may obtain a copy of the License at                                   *
 *                                                                           *
 *     http://www.apache.org/licenses/LICENSE-2.0                            *
 *                                                                           *
 * Unless required by applicable law or agreed to in writing, software       *
 * distributed under the License is distributed on an "AS IS" BASIS,         *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
 * See the License for the specific language governing permissions and       *
 * limitations under the License.                                            *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <openbr/plugins/openbr_internal.h>
#include <openbr/core/qtutils.h>
#include <openbr/core/opencvutils.h>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}

// http://stackoverflow.com/questions/24057248/ffmpeg-undefined-references-to-av-frame-alloc
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55,28,1)
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#endif

using namespace cv;

namespace br
{

/*!
 * \ingroup galleries
 * \brief Read key frames of a video with LibAV
 * \author Ben Klein \cite bhklein
 */
class keyframesGallery : public Gallery
{
    Q_OBJECT

public:
    int64_t idx;

    keyframesGallery()
    {
        av_register_all();
        avformat_network_init();
        avFormatCtx = NULL;
        avCodecCtx = NULL;
        avSwsCtx = NULL;
        avCodec = NULL;
        frame = NULL;
        cvt_frame = NULL;
        buffer = NULL;
        opened = false;
        streamID = -1;
        fps = 0.f;
        time_base = 0.f;
        idx = 0;
        idxOffset = -1;
    }

    ~keyframesGallery()
    {
        release();
    }

    virtual void deferredInit()
    {
        if (avformat_open_input(&avFormatCtx, QtUtils::getAbsolutePath(file.name).toStdString().c_str(), NULL, NULL) != 0) {
            qFatal("Failed to open %s for reading.", qPrintable(file.name));
        } else if (avformat_find_stream_info(avFormatCtx, NULL) < 0) {
            qFatal("Failed to read stream info for %s.", qPrintable(file.name));
        } else {
            for (unsigned int i=0; i<avFormatCtx->nb_streams; i++) {
                if (avFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
                    streamID = i;
                    break;
                }
            }
        }

        if (streamID == -1)
            qFatal("Failed to find video stream for %s", qPrintable(file.name));

        avCodecCtx = avFormatCtx->streams[streamID]->codec;
        avCodec = avcodec_find_decoder(avCodecCtx->codec_id);
        if (avCodec == NULL)
            qFatal("Unsupported codec for %s!", qPrintable(file.name));

        if (avcodec_open2(avCodecCtx, avCodec, NULL) < 0)
            qFatal("Could not open codec for file %s", qPrintable(file.name));

        frame = av_frame_alloc();
        cvt_frame = av_frame_alloc();

        av_init_packet(&packet);
        packet.data = NULL;
        packet.size = 0;
        // Get fps, stream time_base and allocate space for frame buffer with av_malloc.
        fps = (float)avFormatCtx->streams[streamID]->avg_frame_rate.num /
              (float)avFormatCtx->streams[streamID]->avg_frame_rate.den;
        time_base = (float)avFormatCtx->streams[streamID]->time_base.num /
                    (float)avFormatCtx->streams[streamID]->time_base.den;
        int framebytes = avpicture_get_size(AV_PIX_FMT_BGR24, avCodecCtx->width, avCodecCtx->height);
        buffer = (uint8_t*)av_malloc(framebytes*sizeof(uint8_t));
        avpicture_fill((AVPicture*)cvt_frame, buffer, AV_PIX_FMT_BGR24, avCodecCtx->width, avCodecCtx->height);

        avSwsCtx = sws_getContext(avCodecCtx->width, avCodecCtx->height,
                                  avCodecCtx->pix_fmt,
                                  avCodecCtx->width, avCodecCtx->height,
                                  AV_PIX_FMT_BGR24,
                                  SWS_BICUBIC,
                                  NULL, NULL, NULL);

        // attempt to seek to first keyframe
        if (av_seek_frame(avFormatCtx, streamID, avFormatCtx->streams[streamID]->start_time, 0) < 0)
            qFatal("Could not seek to beginning keyframe for %s!", qPrintable(file.name));
        avcodec_flush_buffers(avCodecCtx);

        opened = true;
    }

    TemplateList readBlock(bool *done)
    {
        if (!opened) {
            deferredInit();
        }

        Template output;
        output.file = file;


        int ret = 0;
        while (!ret) {
            if (av_read_frame(avFormatCtx, &packet) >= 0) {
                if (packet.stream_index == streamID) {
                    avcodec_decode_video2(avCodecCtx, frame, &ret, &packet);
                    // Use presentation timestamp if available
                    // Otherwise decode timestamp
                    if (frame->pkt_pts != AV_NOPTS_VALUE)
                        idx = frame->pkt_pts;
                    else
                        idx = frame->pkt_dts;

                    av_free_packet(&packet);
                } else {
                    av_free_packet(&packet);
                }
            } else {
                AVPacket empty_packet;
                av_init_packet(&empty_packet);
                empty_packet.data = NULL;
                empty_packet.size = 0;

                avcodec_decode_video2(avCodecCtx, frame, &ret, &empty_packet);
                if (frame->pkt_pts != AV_NOPTS_VALUE)
                    idx = frame->pkt_pts;
                else if (frame->pkt_dts != AV_NOPTS_VALUE)
                    idx = frame->pkt_dts;
                else // invalid frame
                    ret = 0;

                if (!ret) {
                    av_free_packet(&packet);
                    av_free_packet(&empty_packet);
                    release();
                    *done = true;
                    return TemplateList();
                }
            }
        }

        if (idxOffset < 0) {
            idxOffset = idx;
        }
        // Convert from native format
        sws_scale(avSwsCtx,
                  frame->data,
                  frame->linesize,
                  0, avCodecCtx->height,
                  cvt_frame->data,
                  cvt_frame->linesize);

        // Write AVFrame to cv::Mat
        output.m() = Mat(avCodecCtx->height, avCodecCtx->width, CV_8UC3, cvt_frame->data[0]).clone();
        if (output.m().data) {
            if (av_seek_frame(avFormatCtx, streamID, idx+1, 0) < 0)
                *done = true;
            avcodec_flush_buffers(avCodecCtx);

            QString URL = file.get<QString>("URL", file.name);
            output.file.set("URL", URL + "#t=" + QString::number((int)((idx-idxOffset) * time_base)) + "s");
            output.file.set("timestamp", QString::number((int)((idx-idxOffset) * time_base * 1000)));
            output.file.set("frame", QString::number((idx-idxOffset) * time_base * fps));
            TemplateList dst;
            dst.append(output);
            return dst;
        }
        *done = true;
        return TemplateList();
    }

    void release()
    {
        if (avSwsCtx)     sws_freeContext(avSwsCtx);
        if (frame)        av_free(frame);
        if (cvt_frame)    av_free(cvt_frame);
        if (avCodecCtx)   avcodec_close(avCodecCtx);
        if (avFormatCtx)  avformat_close_input(&avFormatCtx);
        if (buffer)       av_free(buffer);
        avFormatCtx = NULL;
        avCodecCtx = NULL;
        avSwsCtx = NULL;
        avCodec = NULL;
        frame = NULL;
        cvt_frame = NULL;
        buffer = NULL;
    }

    void write(const Template &t)
    {
        (void)t; qFatal("Not implemented");
    }

protected:
    AVFormatContext *avFormatCtx;
    AVCodecContext *avCodecCtx;
    SwsContext *avSwsCtx;
    AVCodec *avCodec;
    AVFrame *frame;
    AVFrame *cvt_frame;
    AVPacket packet;
    uint8_t *buffer;

    int64_t idxOffset;
    bool opened;
    int streamID;
    float fps;
    float time_base;
};

BR_REGISTER(Gallery,keyframesGallery)

/*!
 * \ingroup galleries
 * \brief Read key frames of a .mp4 video file with LibAV
 * \author Ben Klein \cite bhklein
 */
class mp4Gallery : public keyframesGallery
{
    Q_OBJECT
};

BR_REGISTER(Gallery, mp4Gallery)

} // namespace br

#include "gallery/keyframes.moc"
