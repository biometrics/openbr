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
#include "openbr/plugins/openbr_internal.h"
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
        Globals->abbreviations.insert("PP5","Open+Expand+PP5Enroll!PP5Gallery");
        Globals->abbreviations.insert("PP5Register", "PP5Enroll(true,true,0.02,5,Extended)+RenameFirst([eyeL,PP5_Landmark0_Right_Eye],Affine_0)+RenameFirst([eyeR,PP5_Landmark1_Left_Eye],Affine_1)");
        Globals->abbreviations.insert("PP5CropFace", "Open+PP5Enroll(true)+RenameFirst([eyeL,PP5_Landmark0_Right_Eye],Affine_0)+RenameFirst([eyeR,PP5_Landmark1_Left_Eye],Affine_1)+Affine(128,128,0.25,0.35)+Cvt(Gray)");
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

    PP5Context(bool detectOnly = false, float adaptiveMinSize = 0.01f, int minSize = 4, ppr_landmark_range_type landmarkRange = PPR_LANDMARK_RANGE_COMPREHENSIVE, int searchPruningAggressiveness = 0)
    {
        ppr_settings_type default_settings = ppr_get_default_settings();

        default_settings.detection.adaptive_max_size = 1.f;
        default_settings.detection.adaptive_min_size = adaptiveMinSize;
        default_settings.detection.detect_best_face_only = !Globals->enrollAll;
        default_settings.detection.enable = 1;
        default_settings.detection.min_size = minSize;
        default_settings.detection.search_pruning_aggressiveness = searchPruningAggressiveness;
        default_settings.detection.use_serial_face_detection = 1;

        default_settings.landmarks.enable = 1;
        default_settings.landmarks.landmark_range = landmarkRange;
        default_settings.landmarks.manually_detect_landmarks = 0;

        default_settings.recognition.automatically_extract_templates = !detectOnly;
        default_settings.recognition.enable_comparison = !detectOnly;
        default_settings.recognition.enable_extraction = !detectOnly;
        default_settings.recognition.num_comparison_threads = 1;
        default_settings.recognition.recognizer = PPR_RECOGNIZER_MULTI_POSE;
        TRY(ppr_initialize_context(default_settings, &context))
    }

    ~PP5Context()
    {
        TRY(ppr_finalize_context(context))
    }

    static void createRawImage(const cv::Mat &src, ppr_raw_image_type &dst)
    {
        if      (!src.isContinuous()) qFatal("PP5Context::createRawImage requires continuous data.");
        else if (src.channels() == 3) ppr_raw_image_create(&dst, src.cols, src.rows, PPR_RAW_IMAGE_BGR24);
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
        metadata.insert("FrontalFace", QRectF(face_attributes.position.x - face_attributes.dimensions.width/2,
                                              face_attributes.position.y - face_attributes.dimensions.height/2,
                                              face_attributes.dimensions.width,
                                              face_attributes.dimensions.height));
        metadata.insert("Confidence", face_attributes.confidence);
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

    void compareNative(ppr_gallery_type target, const QList<int> &targetIDs, ppr_gallery_type query, const QList<int> &queryIDs, Output *output) const
    {
        ppr_similarity_matrix_type simmat;
        TRY(ppr_compare_galleries(context, query, target, &simmat))
        for (int i=0; i<queryIDs.size(); i++) {
            int query_subject_id = queryIDs[i];
            for (int j=0; j<targetIDs.size(); j++) {
                int target_subject_id = targetIDs[j];
                float score = -std::numeric_limits<float>::max();
                if ((query_subject_id != -1) && (target_subject_id != -1)) {
                    TRY(ppr_get_subject_similarity_score(context, simmat, query_subject_id, target_subject_id, &score))
                }
                output->setRelative(score, i, j);
            }
        }
        ppr_free_similarity_matrix(simmat);
    }

    void enroll(const TemplateList &templates, ppr_gallery_type *gallery, QList<int> &subject_ids) const
    {
        int subject_id = 0, face_id = 0;
        foreach (const Template &src, templates) {
            if (!src.empty() && src.m().data) {
                foreach (const cv::Mat &m, src) {
                    ppr_face_type face;
                    createFace(m, &face);
                    TRY(ppr_add_face(context, gallery, face, subject_id, face_id))
                    face_id++;
                    ppr_free_face(face);
                }
                subject_ids.append(subject_id);
                subject_id++;
            } else {
                subject_ids.append(-1);
            }
        }
    }

};

/*!
 * \ingroup transforms
 * \brief Enroll faces in PP5
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 * \br_property bool detectOnly If true, enroll all detected faces. Otherwise, only enroll faces suitable for recognition. Default is false.
 * \br_property bool requireLandmarks If true, require the right eye, left eye, and nose base to be detectable by PP5. If this does not happen FTE is set to true for that template. Default is false.
 * \br_property float adaptiveMinSize The minimum face size as a percentage of total image width. 0.1 corresponds to a minimum face size of 10% the total image width. Default is 0.01.
 * \br_property int minSize The absolute minimum face size to search for. This is not a pixel value. Please see PittPatt documentation for the relationship between minSize and pixel IPD. Default is 4.
 * \br_property enum landmarkRange Range of landmarks to search for. Options are Frontal, Extended, Full, and Comprehensive. Default is Comprehensive.
 * \br_property int searchPruningAggressiveness The amount of aggressiveness involved in search for faces in images. 0 means all scales and locations are searched. 1 means fewer detectors are used in the early stages but all scales are still searched. 2-4 means that the largest faces are found first and then fewer scales are searched. Default is 0.
 */
class PP5EnrollTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(bool detectOnly READ get_detectOnly WRITE set_detectOnly RESET reset_detectOnly STORED false)
    Q_PROPERTY(bool requireLandmarks READ get_requireLandmarks WRITE set_requireLandmarks RESET reset_requireLandmarks STORED false)
    Q_PROPERTY(float adaptiveMinSize READ get_adaptiveMinSize WRITE set_adaptiveMinSize RESET reset_adaptiveMinSize STORED false)
    Q_PROPERTY(int minSize READ get_minSize WRITE set_minSize RESET reset_minSize STORED false)
    Q_PROPERTY(LandmarkRange landmarkRange READ get_landmarkRange WRITE set_landmarkRange RESET reset_landmarkRange STORED false)
    Q_PROPERTY(int searchPruningAggressiveness READ get_searchPruningAggressiveness WRITE set_searchPruningAggressiveness RESET reset_searchPruningAggressiveness STORED false)

public:
    enum LandmarkRange
    {
        Frontal       = PPR_LANDMARK_RANGE_FRONTAL,
        Extended      = PPR_LANDMARK_RANGE_EXTENDED,
        Full          = PPR_LANDMARK_RANGE_FULL,
        Comprehensive = PPR_LANDMARK_RANGE_COMPREHENSIVE
    };
    Q_ENUMS(LandmarkRange)

private:
    BR_PROPERTY(bool, detectOnly, false)
    BR_PROPERTY(bool, requireLandmarks, false)
    BR_PROPERTY(float, adaptiveMinSize, 0.01f)
    BR_PROPERTY(int, minSize, 4)
    BR_PROPERTY(LandmarkRange, landmarkRange, Comprehensive)
    BR_PROPERTY(int, searchPruningAggressiveness, 0)

    Resource<PP5Context> contexts;

    struct PP5ContextMaker : public ResourceMaker<PP5Context>
    {
        PP5ContextMaker(PP5EnrollTransform *pp5EnrollTransform)
            : pp5EnrollTransform(pp5EnrollTransform) {}

    private:
        PP5EnrollTransform *pp5EnrollTransform;

        PP5Context *make() const
        {
            return new PP5Context(pp5EnrollTransform->detectOnly,
                                  pp5EnrollTransform->adaptiveMinSize,
                                  pp5EnrollTransform->minSize,
                                  (ppr_landmark_range_type) pp5EnrollTransform->landmarkRange,
                                  pp5EnrollTransform->searchPruningAggressiveness);
        }
    };

    void init()
    {
        contexts.setResourceMaker(new PP5ContextMaker(this));
    }

    void project(const Template &src, Template &dst) const
    {
        if (Globals->enrollAll)
            qFatal("single template project doesn't support enrollAll");

        TemplateList srcList;
        srcList.append(src);
        TemplateList dstList;
        project(srcList, dstList);
        dst = dstList.first();
    }

    void project(const TemplateList &srcList, TemplateList &dstList) const
    {
        // Nothing to do here
        if (srcList.empty())
            return;

        PP5Context *context = contexts.acquire();

        foreach (const Template &src, srcList) {
            bool foundFace = false;
            if (!src.isEmpty()) {
                ppr_raw_image_type raw_image;
                PP5Context::createRawImage(src, raw_image);
                ppr_image_type image;
                TRY(ppr_create_image(raw_image, &image))
                ppr_face_list_type face_list;
                TRY(ppr_detect_faces(context->context, image, &face_list))

                for (int i=0; i<face_list.length; i++) {
                    ppr_face_type face = face_list.faces[i];
                    if (!detectOnly) {
                        int extractable;
                        TRY(ppr_is_template_extractable(context->context, face, &extractable))
                        if (!extractable)
                            continue;
                    }
                    foundFace = true;

                    cv::Mat m;
                    if (detectOnly) {
                        m = src;
                    } else {
                        TRY(ppr_extract_face_template(context->context, image, &face))
                        context->createMat(face, m);
                    }
                    Template dst;
                    dst.file = src.file;

                    dst.file.append(PP5Context::toMetadata(face));
                    if (requireLandmarks) {
                        QPointF right = dst.file.get<QPointF>("PP5_Landmark0_Right_Eye");
                        QPointF left = dst.file.get<QPointF>("PP5_Landmark1_Left_Eye");
                        QPointF nose = dst.file.get<QPointF>("PP5_Landmark2_Nose_Base");
                        // a number not equaling itself means it's NaN
                        // there should be no NaNs for the 3 special landmarks
                        if (dst.file.get<int>("PP5_Face_NumLandmarks") < 3 ||
                            right.x() != right.x() || right.y() != right.y() ||
                            left.x() != left.x() || left.y() != left.y() ||
                            nose.x() != nose.x() || nose.y() != nose.y())
                        {
                            dst.file.fte = true;
                        }
                    }
                    dst += m;
                    dstList.append(dst);

                    // Found a face, nothing else to do (if we aren't trying to find multiple faces).
                    if (!Globals->enrollAll)
                        break;
                }

                ppr_free_face_list(face_list);
                ppr_free_image(image);
                ppr_raw_image_free(raw_image);
            }

            // No faces were detected when we were expecting one, output something with FTE set.
            if (!foundFace && !Globals->enrollAll) {
                dstList.append(Template(src.file, detectOnly ? src.m() : cv::Mat()));
                dstList.last().file.fte = true;
            }
        }

        contexts.release(context);
    }
};

BR_REGISTER(Transform, PP5EnrollTransform)


/*!
 * \ingroup distances
 * \brief Compare templates with PP5. PP5 distance is known to be asymmetric
 * \author Josh Klontz \cite jklontz
 * \author E. Taborsky \cite mmtaborsky
 */
class PP5CompareDistance : public UntrainableDistance
                         , public PP5Context
{
    Q_OBJECT

    struct NativeGallery
    {
        FileList files;
        QList<int> subjectIDs;
        ppr_gallery_type gallery;
    };

    mutable QMap<QString, NativeGallery> cache;
    mutable QMutex cacheLock;

    ~PP5CompareDistance()
    {
        foreach (const NativeGallery &gallery, cache.values())
            ppr_free_gallery(gallery.gallery);
    }

    float compare(const cv::Mat &target, const cv::Mat &query) const
    {
        return compare(Template(target), Template(query));
    }

    float compare(const Template &target, const Template &query) const
    {
        TemplateList targetList;
        targetList.append(target);
        TemplateList queryList;
        queryList.append(query);
        MatrixOutput *score = MatrixOutput::make(targetList.files(), queryList.files());
        compare(targetList, queryList, score);
        return score->data.at<float>(0);
    }

    void compare(const TemplateList &target, const TemplateList &query, Output *output) const
    {
        ppr_gallery_type target_gallery, query_gallery;
        ppr_create_gallery(context, &target_gallery);
        ppr_create_gallery(context, &query_gallery);
        QList<int> target_subject_ids, query_subject_ids;
        enroll(target, &target_gallery, target_subject_ids);
        enroll(query, &query_gallery, query_subject_ids);
        compareNative(target_gallery, target_subject_ids, query_gallery, query_subject_ids, output);
        ppr_free_gallery(target_gallery);
        ppr_free_gallery(query_gallery);
    }

    NativeGallery cacheRetain(const File &gallery) const
    {
        QMutexLocker locker(&cacheLock);
        NativeGallery nativeGallery;
        if (cache.contains(gallery.name)) {
            nativeGallery = cache[gallery.name];
        } else {
            ppr_create_gallery(context, &nativeGallery.gallery);
            TemplateList templates = TemplateList::fromGallery(gallery);
            enroll(templates, &nativeGallery.gallery, nativeGallery.subjectIDs);
            nativeGallery.files = templates.files();
            if (gallery.get<bool>("retain"))
                cache.insert(gallery.name, nativeGallery);
        }
        return nativeGallery;
    }

    void cacheRelease(const File &gallery, const NativeGallery &nativeGallery) const
    {
        QMutexLocker locker(&cacheLock);
        if (cache.contains(gallery.name)) {
            if (gallery.get<bool>("release")) {
                cache.remove(gallery.name);
                ppr_free_gallery(nativeGallery.gallery);
            }
        } else {
            ppr_free_gallery(nativeGallery.gallery);
        }
    }

    bool compare(const File &targetGallery, const File &queryGallery, const File &output) const
    {
        if (!targetGallery.get<bool>("native") || !queryGallery.get<bool>("native"))
            return false;

        NativeGallery nativeTarget = cacheRetain(targetGallery);
        NativeGallery nativeQuery = cacheRetain(queryGallery);

        QScopedPointer<Output> o(Output::make(output, nativeTarget.files, nativeQuery.files));
        o->setBlock(0, 0);
        compareNative(nativeTarget.gallery, nativeTarget.subjectIDs, nativeQuery.gallery, nativeQuery.subjectIDs, o.data());

        cacheRelease(targetGallery, nativeTarget);
        cacheRelease(queryGallery, nativeQuery);
        return true;
    }
};

BR_REGISTER(Distance, PP5CompareDistance)

/*!
 * \brief DOCUMENT ME
 * \author Unknown \cite unknown
 */
class PP5GalleryTransform: public UntrainableMetaTransform
                         , public PP5Context
{
    Q_OBJECT
    Q_PROPERTY(QString galleryName READ get_galleryName WRITE set_galleryName RESET reset_galleryName STORED false)
    BR_PROPERTY(QString, galleryName, "")

    ppr_gallery_type target;
    QList<int> targetIDs;
    TemplateList gallery;

    void project(const Template &src, Template &dst) const
    {
        TemplateList temp, output;
        temp.append(src);
        project(temp, output);
        if (!output.empty())
           dst = output[0];
    }

    void project(const TemplateList &src, TemplateList &dst) const
    {
        dst.clear();
        QList<int> queryIDs;

        ppr_gallery_type query;
        ppr_create_gallery(context, &query);
        enroll(src,&query, queryIDs);
        
        ppr_similarity_matrix_type simmat;
        
        TRY(ppr_compare_galleries(context, query, target, &simmat))

        for (int i=0; i<queryIDs.size(); i++) {
            dst.append(Template());
            dst[i].file = src[i].file;
            dst[i].m() = cv::Mat(1,targetIDs.size(), CV_32FC1);

            int query_subject_id = queryIDs[i];
            for (int j=0; j<targetIDs.size(); j++) {
                int target_subject_id = targetIDs[j];
                float score = -std::numeric_limits<float>::max();
                if ((query_subject_id != -1) && (target_subject_id != -1)) {
                    TRY(ppr_get_subject_similarity_score(context, simmat, query_subject_id, target_subject_id, &score))
                }
                dst[i].m().at<float>(0,j) = score;
            }
        }

        ppr_free_similarity_matrix(simmat);
        ppr_free_gallery(query);
    }

    void init()
    {
        if (!galleryName.isEmpty() || !gallery.isEmpty()) {
            // set up the gallery
            ppr_create_gallery(context, &target);
            if (gallery.isEmpty() )
                gallery = TemplateList::fromGallery(galleryName);
            enroll(gallery, &target, targetIDs);
        }
    }

    void train(const TemplateList &data)
    {
        gallery = data;
    }

    void store(QDataStream &stream) const
    {
        br::Object::store(stream);
        stream << gallery;
    }

    void load(QDataStream &stream)
    {
        br::Object::load(stream);
        stream >> gallery;
        init();
    }

};

BR_REGISTER(Transform, PP5GalleryTransform)

#include "classification/pp5.moc"
