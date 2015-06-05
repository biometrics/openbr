#include <QThreadPool>
#include <QMap>
#include <QVariant>
#include <pittpatt_errors.h>
#include <pittpatt_nc_sdk.h>
#include <pittpatt_raw_image_io.h>
#include <pittpatt_license.h>
#include <openbr/openbr_plugin.h>
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
 * \brief Initialize PittPatt 4
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer.
 */
class PP4Initializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        Globals->abbreviations.insert("PP4", "Open+PP4Enroll:PP4Compare");
    }

    void finalize() const
    {
        ppr_finalize_sdk();
    }
};

BR_REGISTER(Initializer, PP4Initializer)

/*!
 * \brief PittPatt 4 context
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer.
 */
struct PP4Context
{
    ppr_context_type context;

    PP4Context()
    {
        context = ppr_get_context();
        TRY(ppr_enable_recognition(context))
        TRY(ppr_set_license(context, my_license_id, my_license_key))
        TRY(ppr_set_models_path(context, qPrintable(Globals->sdkPath + "/models/pp4")))
        TRY(ppr_set_num_recognition_threads(context, QThreadPool::globalInstance()->maxThreadCount()))
        TRY(ppr_set_num_detection_threads(context, 1))
        TRY(ppr_set_detection_precision(context, PPR_FINE_PRECISION))
        TRY(ppr_set_landmark_detector_type(context, PPR_DUAL_MULTI_POSE_LANDMARK_DETECTOR, PPR_AUTOMATIC_LANDMARKS))
        TRY(ppr_set_min_size(context, 4))
        TRY(ppr_set_frontal_yaw_constraint(context, PPR_FRONTAL_YAW_CONSTRAINT_PERMISSIVE))
        TRY(ppr_set_template_extraction_type(context, PPR_EXTRACT_DOUBLE))
        TRY(ppr_initialize_context(context))
    }

    ~PP4Context()
    {
        TRY(ppr_release_context(context))
    }

    static void createRawImage(const cv::Mat &src, ppr_raw_image_type &dst)
    {
        ppr_raw_image_create(&dst, src.cols, src.rows, PPR_RAW_IMAGE_BGR24);
        assert((src.type() == CV_8UC3) && src.isContinuous());
        memcpy(dst.data, src.data, 3*src.rows*src.cols);
    }

    void createMat(const ppr_template_type &src, cv::Mat &dst) const
    {
        ppr_flat_template_type flat_template;
        TRY(ppr_flatten_template(context,src,&flat_template))
        dst = cv::Mat(1, flat_template.num_bytes, CV_8UC1, flat_template.data).clone();
        ppr_free_flat_template(flat_template);
    }

    void createTemplate(const cv::Mat &src, ppr_template_type *dst) const
    {
        ppr_flat_template_type flat_template;
        flat_template.num_bytes = src.cols;
        flat_template.data = src.data;
        TRY(ppr_unflatten_template(context, flat_template, dst))
    }

    static QString toString(const ppr_landmark_category_type &category)
    {
        switch (category) {
          case PPR_LANDMARK_LEFT_EYE:
            return "Left_Eye";
          case PPR_LANDMARK_RIGHT_EYE:
            return "Right_Eye";
          case PPR_LANDMARK_NOSE_BASE:
            return "Nose_Base";
          case PPR_LANDMARK_NOSE_BRIDGE:
            return "Nose_Bridge";
          case PPR_LANDMARK_NOSE_TIP:
            return "Nose_Tip";
          case PPR_LANDMARK_NOSE_TOP:
            return "Nose_Top";
          case PPR_LANDMARK_EYE_NOSE:
            return "Eye_Nose";
          case PPR_LANDMARK_MOUTH:
            return "Mouth";
        }

        return "Unknown";
    }

    static QMap<QString,QVariant> toMetadata(const ppr_object_type &object)
    {
        QMap<QString,QVariant> metadata;

        metadata.insert("FrontalFace", QRectF(object.position.x - object.dimensions.width/2,
                                              object.position.y - object.dimensions.height/2,
                                              object.dimensions.width,
                                              object.dimensions.height));
        metadata.insert("Confidence", object.confidence);
        metadata.insert("PP4_Object_X", object.position.x - object.dimensions.width/2);
        metadata.insert("PP4_Object_Y", object.position.y - object.dimensions.height/2);
        metadata.insert("PP4_Object_Width", object.dimensions.width);
        metadata.insert("PP4_Object_Height", object.dimensions.height);
        metadata.insert("PP4_Object_Roll", object.rotation.roll);
        metadata.insert("PP4_Object_Pitch", object.rotation.pitch);
        metadata.insert("PP4_Object_Yaw", object.rotation.yaw);
        metadata.insert("PP4_Object_Precision", object.rotation.precision);
        metadata.insert("PP4_Object_ModelID", object.model_id);
        metadata.insert("PP4_Object_NumLandmarks", object.num_landmarks);
        metadata.insert("PP4_Object_Size", object.size);

        QList<ppr_landmark_category_type> categories;
        categories << PPR_LANDMARK_RIGHT_EYE
                   << PPR_LANDMARK_LEFT_EYE
                   << PPR_LANDMARK_NOSE_BASE
                   << PPR_LANDMARK_NOSE_BRIDGE
                   << PPR_LANDMARK_NOSE_TIP
                   << PPR_LANDMARK_NOSE_TOP
                   << PPR_LANDMARK_EYE_NOSE
                   << PPR_LANDMARK_MOUTH;

        for (int i=0; i<categories.size(); i++) {
            ppr_landmark_category_type category = categories[i];
            QString metadataString = QString("PP4_Landmark%1_%2").arg(QString::number(i), toString(category));

            bool found = false;
            for (int j=0; j<object.num_landmarks; j++) {
                ppr_landmark_type &landmark = object.landmarks[j];
                if (landmark.category != category) continue;

                metadata.insert(metadataString+"_X", landmark.position.x);
                metadata.insert(metadataString+"_Y", landmark.position.y);
                metadata.insert(metadataString+"_Category", landmark.category);
                metadata.insert(metadataString+"_ModelID", landmark.model_id);
                metadata.insert(metadataString+"_Index", j);
                found = true;
                break;
            }

            if (!found) {
                metadata.insert(metadataString+"_X", -1);
                metadata.insert(metadataString+"_Y", -1);
                metadata.insert(metadataString+"_Category", -1);
                metadata.insert(metadataString+"_ModelID", -1);
                metadata.insert(metadataString+"_Index", -1);
            }
        }

        return metadata;
    }

    static void freeObject(ppr_object_type &object)
    {
        delete[] object.landmarks;
        object.landmarks = NULL;
        object.num_landmarks = 0;
    }
};

/*!
 * \ingroup transforms
 * \brief Enroll faces in PittPatt 4
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer.
 * \br_property bool detectOnly If true, return all detected faces. Otherwise, return only faces that are suitable for recognition. Default is false.
 */
class PP4EnrollTransform : public UntrainableMetaTransform
{
    Q_OBJECT
    Q_PROPERTY(bool detectOnly READ get_detectOnly WRITE set_detectOnly RESET reset_detectOnly STORED false)
    BR_PROPERTY(bool, detectOnly, false)
    Resource<PP4Context> contexts;

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
        if (srcList.empty())
            return;

        PP4Context *context = contexts.acquire();

        foreach(const Template &src, srcList) {
            if (!src.isEmpty()) {
                ppr_raw_image_type raw_image;
                PP4Context::createRawImage(src, raw_image);
                ppr_image_type image;
                TRY(ppr_create_image(raw_image, &image))
                ppr_object_list_type object_list;
                TRY(ppr_detect_objects(context->context, image, &object_list))

                QList<ppr_object_type> objects;
                if (Globals->enrollAll) objects = getAllObjects(object_list);
                else                    objects = getBestObject(context, object_list);

                foreach (const ppr_object_type &object, objects) {
                    ppr_object_suitability_type suitability;
                    TRY(ppr_is_object_suitable_for_recognition(context->context, object, &suitability))
                    if (suitability != PPR_OBJECT_SUITABLE_FOR_RECOGNITION && !detectOnly) continue;

                    cv::Mat m;
                    if (detectOnly)
                        m = src;
                    else {
                        ppr_template_type curr_template;
                        TRY(ppr_extract_template_from_object(context->context, image, object, &curr_template))
                        context->createMat(curr_template, m);
                    }

                    Template dst;
                    dst.file = src.file;

                    dst.file.append(PP4Context::toMetadata(object));
                    dst += m;
                    dstList.append(dst);

                    if (!Globals->enrollAll)
                        break;
                }

                ppr_free_object_list(object_list);
                ppr_free_image(image);
                ppr_raw_image_free(raw_image);
            }

            if (!Globals->enrollAll && dstList.empty()) {
                dstList.append(Template(src.file, detectOnly ? src.m() : cv::Mat()));
                dstList.last().file.fte = true;
            }
        }

        contexts.release(context);
    }

private:
    QList<ppr_object_type> getBestObject(PP4Context *context, ppr_object_list_type object_list) const
    {
        int best_index = -1;
        float best_confidence = 0;
        for (int i=0; i<object_list.num_objects; i++) {
            ppr_object_type object = object_list.objects[i];
            ppr_object_suitability_type suitability;
            TRY(ppr_is_object_suitable_for_recognition(context->context, object, &suitability))
            if (suitability != PPR_OBJECT_SUITABLE_FOR_RECOGNITION) continue;
            if ((object.confidence > best_confidence) ||
                (best_index == -1)) {
                best_confidence = object.confidence;
                best_index = i;
            }
        }

        QList<ppr_object_type> objects;
        if (best_index != -1) objects.append(object_list.objects[best_index]);
        return objects;
    }

    QList<ppr_object_type> getAllObjects(ppr_object_list_type object_list) const
    {
        QList<ppr_object_type> objects;
        for (int i=0; i<object_list.num_objects; i++)
            objects.append(object_list.objects[i]);
        return objects;
    }
};

BR_REGISTER(Transform, PP4EnrollTransform)

/*!
 * \ingroup distances
 * \brief Compare faces using PittPatt 4.
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer.
 */
class PP4Compare : public Distance,
                   public PP4Context
{
    Q_OBJECT

    void compare(const TemplateList &target, const TemplateList &query, Output *output) const
    {
        ppr_gallery_type target_gallery, query_gallery;
        ppr_create_gallery(context, &target_gallery);
        ppr_create_gallery(context, &query_gallery);
        QList<int> target_template_ids, query_template_ids;
        enroll(target, &target_gallery, target_template_ids);
        enroll(query, &query_gallery, query_template_ids);

        ppr_similarity_matrix_type similarity_matrix;
        TRY(ppr_compare_galleries(context, query_gallery, target_gallery, &similarity_matrix))

        for (int i=0; i<query_template_ids.size(); i++) {
            int query_template_id = query_template_ids[i];
            for (int j=0; j<target_template_ids.size(); j++) {
                int target_template_id = target_template_ids[j];
                float score = -std::numeric_limits<float>::max();
                if ((query_template_id != -1) && (target_template_id != -1)) {
                    TRY(ppr_get_similarity_matrix_element(context, similarity_matrix, query_template_id, target_template_id, &score))
                }
                output->setRelative(score, i, j);
            }
        }

        ppr_free_similarity_matrix(similarity_matrix);
        ppr_free_gallery(target_gallery);
        ppr_free_gallery(query_gallery);
    }

    void enroll(const TemplateList &templates, ppr_gallery_type *gallery, QList<int> &template_ids) const
    {
        foreach (const Template &t, templates) {
            if (t.m().data) {
                ppr_template_type u;
                createTemplate(t.m(), &u);
                int template_id;
                TRY(ppr_copy_template_to_gallery(context, gallery, u, &template_id))
                template_ids.append(template_id);
                ppr_free_template(u);
            } else {
                template_ids.append(-1);
            }
        }
    }
};

BR_REGISTER(Distance, PP4Compare)

#include "plugins/pp4.moc"
