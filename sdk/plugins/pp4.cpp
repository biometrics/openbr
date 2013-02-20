#include <QThreadPool>
#include <pittpatt_errors.h>
#include <pittpatt_nc_sdk.h>
#include <pittpatt_raw_image_io.h>
#include <pittpatt_license.h>
#include <mm_plugin.h>

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

using namespace mm;

/*!
 * \brief PittPatt 4 context
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer.
 */
struct PP4Context
{
    static ppr_context_type context;

    static void createRawImage(const cv::Mat &src, ppr_raw_image_type &dst)
    {
        ppr_raw_image_create(&dst, src.cols, src.rows, PPR_RAW_IMAGE_BGR24);
        assert((src.type() == CV_8UC3) && src.isContinuous());
        memcpy(dst.data, src.data, 3*src.rows*src.cols);
    }

    static void createMat(const ppr_template_type &src, cv::Mat &dst)
    {
        ppr_flat_template_type flat_template;
        TRY(ppr_flatten_template(context,src,&flat_template))
        dst = cv::Mat(1, flat_template.num_bytes, CV_8UC1, flat_template.data).clone();
        ppr_free_flat_template(flat_template);
    }

    static void createTemplate(const cv::Mat &src, ppr_template_type *dst)
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

    static File toMetadata(const ppr_object_type &object)
    {
        File metadata;
        metadata.insert("PP4_Object_X", object.position.x - object.dimensions.width/2);
        metadata.insert("PP4_Object_Y", object.position.y - object.dimensions.height/2);
        metadata.insert("PP4_Object_Width", object.dimensions.width);
        metadata.insert("PP4_Object_Height", object.dimensions.height);
        metadata.insert("PP4_Object_Confidence", object.confidence);
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

    static ppr_object_type fromMetadata(const File &metadata)
    {
        ppr_object_type object;

        object.position.x = metadata.value("PP4_Object_X").toFloat() + metadata.value("PP4_Object_Width").toFloat()/2;
        object.position.y = metadata.value("PP4_Object_Y").toFloat() + metadata.value("PP4_Object_Height").toFloat()/2;
        object.dimensions.width = metadata.value("PP4_Object_Width").toFloat();
        object.dimensions.height = metadata.value("PP4_Object_Height").toFloat();
        object.confidence = metadata.value("PP4_Object_Confidence").toFloat();
        object.rotation.roll = metadata.value("PP4_Object_Roll").toFloat();
        object.rotation.pitch = metadata.value("PP4_Object_Pitch").toFloat();
        object.rotation.yaw = metadata.value("PP4_Object_Yaw").toFloat();
        object.rotation.precision = (ppr_precision_type) metadata.value("PP4_Object_Precision").toFloat();
        object.model_id = metadata.value("PP4_Object_ModelID").toInt();
        object.num_landmarks = metadata.value("PP4_Object_NumLandmarks").toInt();
        object.size = metadata.value("PP4_Object_Size").toFloat();

        QStringList landmarkNames = QStringList(metadata.keys()).filter(QRegExp("(.*)_Category")).replaceInStrings("_Category", "");
        object.landmarks = new ppr_landmark_type[object.num_landmarks];
        for (int j=0; j<landmarkNames.size(); j++) {
            int landmarkIndex = metadata.value(landmarkNames[j]+"_Index").toInt();
            if (landmarkIndex == -1) continue;
            object.landmarks[landmarkIndex].position.x = metadata.value(landmarkNames[j]+"_X").toFloat();
            object.landmarks[landmarkIndex].position.y = metadata.value(landmarkNames[j]+"_Y").toFloat();
            object.landmarks[landmarkIndex].category = (ppr_landmark_category_type)metadata.value(landmarkNames[j]+"_Category").toInt();
            object.landmarks[landmarkIndex].model_id = metadata.value(landmarkNames[j]+"_ModelID").toInt();
            landmarkIndex++;
        }

        return object;
    }

    static void freeObject(ppr_object_type &object)
    {
        delete[] object.landmarks;
        object.landmarks = NULL;
        object.num_landmarks = 0;
    }
};

ppr_context_type PP4Context::context;

/*!
 * \ingroup initializers
 * \brief Initialize PittPatt 4
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer.
 */
class PP4Initializer : public Initializer
                     , public PP4Context
{
    Q_OBJECT

    void initialize() const
    {
        context = ppr_get_context();
        TRY(ppr_enable_recognition(context))
        TRY(ppr_set_license(context, my_license_id, my_license_key))
        TRY(ppr_set_models_path(context, qPrintable(Globals->SDKPath + "/models/pp4")))
        TRY(ppr_set_num_recognition_threads(context, QThreadPool::globalInstance()->maxThreadCount()))
        TRY(ppr_set_num_detection_threads(context, 1))
        TRY(ppr_set_detection_precision(context, PPR_FINE_PRECISION))
        TRY(ppr_set_landmark_detector_type(context, PPR_DUAL_MULTI_POSE_LANDMARK_DETECTOR, PPR_AUTOMATIC_LANDMARKS))
        TRY(ppr_set_min_size(context, 4))
        TRY(ppr_set_frontal_yaw_constraint(context, PPR_FRONTAL_YAW_CONSTRAINT_PERMISSIVE))
        TRY(ppr_set_template_extraction_type(context, PPR_EXTRACT_DOUBLE))
        TRY(ppr_initialize_context(context))
        Globals->Abbreviations.insert("PP4", "Open+PP4Detect!PP4Enroll:PP4Compare");
    }

    void finalize() const
    {
        TRY(ppr_release_context(context))
        ppr_finalize_sdk();
    }
};

MM_REGISTER(Initializer, PP4Initializer, "")

/*!
 * \ingroup transforms
 * \brief Detect a face in PittPatt 4
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer.
 */
class PP4Detect : public UntrainableMetaFeature
                , public PP4Context
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        dst.file = src.file;

        foreach (const cv::Mat &matrix, src) {
            ppr_raw_image_type raw_image;
            createRawImage(matrix, raw_image);
            ppr_image_type image;
            TRY(ppr_create_image(raw_image, &image))
            ppr_object_list_type object_list;
            TRY(ppr_detect_objects(context, image, &object_list))

            QList<ppr_object_type> objects;
            if (src.file.getBool("ForceEnrollment")) objects = getBestObject(object_list);
            else                                     objects = getAllObjects(object_list);

            foreach (const ppr_object_type &object, objects) {
                dst.file.append(toMetadata(object));
                dst += matrix;
            }

            ppr_free_object_list(object_list);
            ppr_free_image(image);
            ppr_raw_image_free(raw_image);
        }

        if (src.file.getBool("ForceEnrollment") && dst.isEmpty()) dst += cv::Mat();
    }

private:
    QList<ppr_object_type> getBestObject(ppr_object_list_type object_list) const
    {
        int best_index = -1;
        float best_confidence = 0;
        for (int i=0; i<object_list.num_objects; i++) {
            ppr_object_type object = object_list.objects[i];
            ppr_object_suitability_type suitability;
            TRY(ppr_is_object_suitable_for_recognition(context, object, &suitability))
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

MM_REGISTER(Feature, PP4Detect, "")

/*!
 * \ingroup transforms
 * \brief Enroll face in PittPatt 4
 * \author Josh Klontz \cite jklontz
 * \warning Needs a maintainer.
 */
class PP4Enroll : public UntrainableMetaFeature
                , public PP4Context
{
    Q_OBJECT

    void project(const Template &src, Template &dst) const
    {
        if (!src.m().data) {
            dst += cv::Mat();
            return;
        }

        ppr_raw_image_type raw_image;
        createRawImage(src, raw_image);
        ppr_image_type image;
        TRY(ppr_create_image(raw_image, &image))

        ppr_object_type object = fromMetadata(src.file);

        ppr_template_type curr_template;
        TRY(ppr_extract_template_from_object(context, image, object, &curr_template))

        freeObject(object);

        cv::Mat m;
        createMat(curr_template, m);
        dst += m;

        ppr_free_template(curr_template);
        ppr_free_image(image);
        ppr_raw_image_free(raw_image);
    }
};

MM_REGISTER(Feature, PP4Enroll, "")


class PP4Compare : public Comparer,
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
                output->setData(score, i, j);
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

MM_REGISTER(Comparer, PP4Compare, "")

#include "plugins/pp4.moc"
