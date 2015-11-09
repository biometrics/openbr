#include "ssm.h"

using namespace cv;
using namespace br;

void scoresMerge(const QMap<int, TemplateList> &subjectMap, TemplateList &subjects)
{
    foreach (const TemplateList &tlist, subjectMap.values()) {
        Template subject;
        foreach (const Template &t, tlist) {
            if (!t.file.fte)
                subject.merge(t);
        }

        if (subject.empty())
            subject.file.fte = true;
        subjects.append(subject);
    }
}

void averageMerge(const QMap<int, TemplateList> &subjectMap, TemplateList &subjects)
{
    foreach (const TemplateList &tlist, subjectMap.values()) {
        File f; Mat m;
        foreach (const Template &t, tlist) {
            if (t.file.fte)
                continue;

            f.append(t.file);
            if (m.empty())
                m = Mat::zeros(t.m().rows, t.m().cols, CV_MAKETYPE(CV_32F, t.m().channels()));
            m += t.m();
        }
        m /= tlist.size();

        Template subject(f, m);
        subjects.append(subject);
    }
}

void frontalMerge(const QMap<int, TemplateList> &subjectMap, TemplateList &subjects)
{
    foreach (const TemplateList &tlist, subjectMap.values()) {
        Template subject; float yaw = std::numeric_limits<float>::max();
        foreach (const Template &t, tlist) {
            if (fabs(t.file.get<float>("Yaw")) < yaw) {
                yaw = fabs(t.file.get<float>("Yaw"));
                subject = t;
            }
        }
        subjects.append(subject);
    }
}

void br::SSM(const QString &image_gallery, const QString &subject_gallery, const QString &method)
{
    const TemplateList images = TemplateList::fromGallery(image_gallery);

    QMap<int, TemplateList> subjectMap;
    foreach (const Template &t, images) {
        int templateID = t.file.get<int>("TemplateID");
        if (subjectMap.contains(templateID)) subjectMap[templateID].append(t);
        else                                 subjectMap[templateID] = TemplateList() << t;
    }

    TemplateList subjects;
    if (method == "Scores")
        scoresMerge(subjectMap, subjects);
    else if (method == "Average")
        averageMerge(subjectMap, subjects);
    else if (method == "Frontal")
        frontalMerge(subjectMap, subjects);
    else
        qFatal("Unknown SSM method %s. Options are Scores|Average|Frontal", method.toStdString().c_str());

    QScopedPointer<Gallery> output(Gallery::make(subject_gallery));
    output->writeBlock(subjects);
}
