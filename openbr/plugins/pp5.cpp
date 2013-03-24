#include <QDebug>
#include <QFileInfo>
#include <QMap>
#include <QMutex>
#include <QSharedPointer>
#include <QString>
#include <QStringList>
#include <QThreadPool>
#include <QVariant>
#include <pittpatt_errors.h>
#include <pittpatt_raw_image_io.h>
#include <pittpatt_sdk.h>
#include <pittpatt_license.h>
#include <openbr/openbr_plugin.h>

#include "openbr/core/resource.h"

#define TRY(CC)                                                                                               \
{                                                                                                             \
    if ((CC) != PPR_SUCCESS) qFatal("%d error (%s, %d): %s.", CC, __FILE__, __LINE__, ppr_error_message(CC)); \
}

#define TRY_VIDEO(CC)                                                                                                           \
{                                                                                                                               \
    if ((CC) != PPR_VIDEO_IO_SUCCESS) qFatal("%d error (%s, %d): %s.", CC, __FILE__, __LINE__, ppr_video_io_error_message(CC)); \
}

#define TRY_RAW_IMAGE(CC)                                                                                                         \
{                                                                                                                                 \
    if ((CC) != PPR_RAW_IMAGE_SUCCESS) qFatal("%d error (%s, %d): %s.", CC, __FILE__, __LINE__, ppr_raw_image_error_message(CC)); \
}

using namespace br;

/*!
 * \ingroup initializers
 * \brief Initialize PP5
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 * \warning PittPatt 5.x.x is known to NOT work with MinGW-w64 due to a segfault in ppr_initialize_sdk.
 */
class PP5Initializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        TRY(ppr_initialize_sdk(qPrintable(Globals->sdkPath + "/share/openbr/models/pp5/"), my_license_id, my_license_key))
        Globals->abbreviations.insert("PP5","Open!PP5Enroll:PP5Compare");
        Globals->abbreviations.insert("PP5Register", "Open+PP5Enroll(true)+RenameFirst([eyeL,PP5_Landmark0_Right_Eye],Affine_0)+RenameFirst([eyeR,PP5_Landmark1_Left_Eye],Affine_1)");
    }

    void finalize() const
    {
        ppr_finalize_sdk();
    }
};

BR_REGISTER(Initializer, PP5Initializer)

/*!
 * \brief PP5 context
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 */
struct PP5Context
{
    ppr_context_type context;

    PP5Context()
    {
        ppr_settings_type default_settings = ppr_get_default_settings();

        default_settings.detection.adaptive_max_size = 1.f;
        default_settings.detection.adaptive_min_size = 0.01f;
        default_settings.detection.detect_best_face_only = true;
        default_settings.detection.enable = 1;
        default_settings.detection.min_size = 4;
        default_settings.detection.use_serial_face_detection = 1;

        default_settings.landmarks.enable = 1;
        default_settings.landmarks.landmark_range = PPR_LANDMARK_RANGE_COMPREHENSIVE;
        default_settings.landmarks.manually_detect_landmarks = 0;

        default_settings.recognition.automatically_extract_templates = 1;
        default_settings.recognition.enable_comparison = 1;
        default_settings.recognition.enable_extraction = 1;
        default_settings.recognition.num_comparison_threads = QThreadPool::globalInstance()->maxThreadCount();
        default_settings.recognition.recognizer = PPR_RECOGNIZER_MULTI_POSE;
        TRY(ppr_initialize_context(default_settings, &context))
    }

    ~PP5Context()
    {
        TRY(ppr_finalize_context(context))
    }

    static void createRawImage(const cv::Mat &src, ppr_raw_image_type &dst)
    {
        if (!src.isContinuous()) qFatal("PP5Context::createRawImage requires continuous data.");
        if      (src.channels() == 3) ppr_raw_image_create(&dst, src.cols, src.rows, PPR_RAW_IMAGE_BGR24);
        else if (src.channels() == 1) ppr_raw_image_create(&dst, src.cols, src.rows, PPR_RAW_IMAGE_GRAY8);
        else                          qFatal("PP5Context::createRawImage invalid channel count.");
        memcpy(dst.data, src.data, src.channels()*src.rows*src.cols);
    }

    void createMat(const ppr_face_type &src, cv::Mat &dst) const
    {
        ppr_flat_data_type flat_data;
        TRY(ppr_flatten_face(context,src,&flat_data))
        dst = cv::Mat(1, flat_data.length, CV_8UC1, flat_data.data).clone();
        ppr_free_flat_data(flat_data);
    }

    void createFace(const cv::Mat &src, ppr_face_type *dst) const
    {
        ppr_flat_data_type flat_data;
        flat_data.length = src.cols;
        flat_data.data = src.data;
        TRY(ppr_unflatten_face(context, flat_data, dst))
    }

    static QString toString(const ppr_landmark_category_type &category)
    {
        switch (category) {
          case PPR_LANDMARK_CATEGORY_LEFT_EYE:
            return "Left_Eye";
          case PPR_LANDMARK_CATEGORY_RIGHT_EYE:
            return "Right_Eye";
          case PPR_LANDMARK_CATEGORY_NOSE_BASE:
            return "Nose_Base";
          case PPR_LANDMARK_CATEGORY_NOSE_BRIDGE:
            return "Nose_Bridge";
          case PPR_LANDMARK_CATEGORY_EYE_NOSE:
            return "Eye_Nose";
          case PPR_LANDMARK_CATEGORY_LEFT_UPPER_CHEEK:
            return "Left_Upper_Cheek";
          case PPR_LANDMARK_CATEGORY_LEFT_LOWER_CHEEK:
            return "Left_Lower_Cheek";
          case PPR_LANDMARK_CATEGORY_RIGHT_UPPER_CHEEK:
            return "Right_Upper_Cheek";
          case PPR_LANDMARK_CATEGORY_RIGHT_LOWER_CHEEK:
            return "Right_Lower_Cheek";
          case PPR_NUM_LANDMARK_CATEGORIES:
            return "Num_Landmark_Categories";
        }

        return "Unknown";
    }

    static QMap<QString,QVariant> toMetadata(const ppr_face_type &face)
    {
        QMap<QString,QVariant> metadata;

        ppr_face_attributes_type face_attributes;
        ppr_get_face_attributes(face, &face_attributes);
        metadata.insert("Face", QRectF(face_attributes.position.x - face_attributes.dimensions.width/2,
                                       face_attributes.position.y - face_attributes.dimensions.height/2,
                                       face_attributes.dimensions.width,
                                       face_attributes.dimensions.height));
        metadata.insert("PP5_Face_Confidence", face_attributes.confidence);
        metadata.insert("PP5_Face_Roll", face_attributes.rotation.roll);
        metadata.insert("PP5_Face_Pitch", face_attributes.rotation.pitch);
        metadata.insert("PP5_Face_Yaw", face_attributes.rotation.yaw);
        metadata.insert("PP5_Face_HasThumbnail", face_attributes.has_thumbnail);
        metadata.insert("PP5_Face_NumLandmarks", face_attributes.num_landmarks);
        metadata.insert("PP5_Face_Size", face_attributes.size);
        metadata.insert("PP5_TrackingInfo_ConfidenceLevel", face_attributes.tracking_info.confidence_level);
        metadata.insert("PP5_TrackingInfo_FrameNumber", face_attributes.tracking_info.frame_number);
        metadata.insert("PP5_TrackingInfo_TrackID", face_attributes.tracking_info.track_id);

        ppr_landmark_list_type landmark_list;
        TRY(ppr_get_face_landmarks(face, &landmark_list))

        QList<ppr_landmark_category_type> categories;
        categories << PPR_LANDMARK_CATEGORY_RIGHT_EYE
                   << PPR_LANDMARK_CATEGORY_LEFT_EYE
                   << PPR_LANDMARK_CATEGORY_NOSE_BASE
                   << PPR_LANDMARK_CATEGORY_NOSE_BRIDGE
                   << PPR_LANDMARK_CATEGORY_EYE_NOSE
                   << PPR_LANDMARK_CATEGORY_LEFT_UPPER_CHEEK
                   << PPR_LANDMARK_CATEGORY_LEFT_LOWER_CHEEK
                   << PPR_LANDMARK_CATEGORY_RIGHT_UPPER_CHEEK
                   << PPR_LANDMARK_CATEGORY_RIGHT_LOWER_CHEEK;
        for (int i=0; i<categories.size(); i++) {
            ppr_landmark_category_type category = categories[i];
            QString metadataString = QString("PP5_Landmark%1_%2").arg(QString::number(i), toString(category));

            bool found = false;
            for (int j=0; j<landmark_list.length; j++) {
                ppr_landmark_type &landmark = landmark_list.landmarks[j];
                if (landmark.category != category) continue;

                metadata.insert(metadataString, QPointF(landmark.position.x, landmark.position.y));
                found = true;
                break;
            }

            if (!found) {
                metadata.insert(metadataString, QPointF(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
            }
        }

        ppr_free_landmark_list(landmark_list);

        return metadata;
    }
};

/*!
 * \ingroup transforms
 * \brief Enroll faces in PP5
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 */
class PP5EnrollTransform : public UntrainableTransform
{
    Q_OBJECT
    Q_PROPERTY(bool detectOnly READ get_detectOnly WRITE set_detectOnly RESET reset_detectOnly STORED false)
    BR_PROPERTY(bool, detectOnly, false)
    Resource<PP5Context> contexts;

    void project(const Template &src, Template &dst) const
    {
        PP5Context *context = contexts.acquire();

        ppr_raw_image_type raw_image;
        PP5Context::createRawImage(src, raw_image);
        ppr_image_type image;
        TRY(ppr_create_image(raw_image, &image))
        ppr_face_list_type face_list;
        TRY(ppr_detect_faces(context->context, image, &face_list))

        for (int i=0; i<face_list.length; i++) {
            ppr_face_type face = face_list.faces[i];
            int extractable;
            TRY(ppr_is_template_extractable(context->context, face, &extractable))
            if (!extractable && !detectOnly) continue;

            cv::Mat m;
            if (detectOnly) {
                m = src;
            } else {
                TRY(ppr_extract_face_template(context->context, image, &face))
                context->createMat(face, m);
            }

            dst.file.append(PP5Context::toMetadata(face));
            dst += m;

            if (!src.file.getBool("enrollAll")) break;
        }

        ppr_free_face_list(face_list);
        ppr_free_image(image);
        ppr_raw_image_free(raw_image);

        contexts.release(context);

        if (!src.file.getBool("enrollAll") && dst.isEmpty()) {
            if (detectOnly) dst += src;
            else            dst += cv::Mat();
        }
    }
};

BR_REGISTER(Transform, PP5EnrollTransform)

/*!
 * \ingroup distances
 * \brief Compare templates with PP5
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 */
class PP5CompareDistance : public Distance
                         , public PP5Context
{
    Q_OBJECT

    float compare(const Template &target, const Template &query) const
    {
        (void) target;
        (void) query;
        qFatal("Compare single templates should never be called!");
        return 0;
    }

    void compare(const TemplateList &target, const TemplateList &query, Output *output) const
    {
        ppr_gallery_type target_gallery, query_gallery;
        ppr_create_gallery(context, &target_gallery);
        ppr_create_gallery(context, &query_gallery);
        QList<int> target_face_ids, query_face_ids;
        enroll(target, &target_gallery, target_face_ids);
        enroll(query, &query_gallery, query_face_ids);

        ppr_similarity_matrix_type similarity_matrix;
        TRY(ppr_compare_galleries(context, query_gallery, target_gallery, &similarity_matrix))

        for (int i=0; i<query_face_ids.size(); i++) {
            int query_face_id = query_face_ids[i];
            for (int j=0; j<target_face_ids.size(); j++) {
                int target_face_id = target_face_ids[j];
                float score = -std::numeric_limits<float>::max();
                if ((query_face_id != -1) && (target_face_id != -1)) {
                    TRY(ppr_get_face_similarity_score(context, similarity_matrix, query_face_id, target_face_id, &score))
                }
                output->setRelative(score, i, j);
            }
        }

        ppr_free_similarity_matrix(similarity_matrix);
        ppr_free_gallery(target_gallery);
        ppr_free_gallery(query_gallery);
    }

    void enroll(const TemplateList &templates, ppr_gallery_type *gallery, QList<int> &face_ids) const
    {
        int face_id = 0;
        foreach (const Template &src, templates) {
            if (src.m().data) {
                ppr_face_type face;
                createFace(src, &face);
                TRY(ppr_add_face(context, gallery, face, face_id, face_id))
                face_ids.append(face_id);
                face_id++;
                ppr_free_face(face);
            } else {
                face_ids.append(-1);
            }
        }
    }
};

BR_REGISTER(Distance, PP5CompareDistance)

#include "plugins/pp5.moc"
