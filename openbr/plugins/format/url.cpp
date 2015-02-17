#include <QtNetwork>
#include <opencv2/highgui/highgui.hpp>

#include <openbr/plugins/openbr_internal.h>

using namespace cv;

namespace br
{

/*!
 * \ingroup formats
 * \brief Reads image files from the web.
 * \author Josh Klontz \cite jklontz
 */
class urlFormat : public Format
{
    Q_OBJECT

    Template read() const
    {
        Template t;

        QNetworkAccessManager networkAccessManager;
        QNetworkRequest request(QString(file.name).remove(".url"));
        request.setAttribute(QNetworkRequest::CacheLoadControlAttribute, QNetworkRequest::AlwaysNetwork);
        QNetworkReply *reply = networkAccessManager.get(request);

        while (!reply->isFinished()) QCoreApplication::processEvents();
        if (reply->error()) qWarning("%s (%s)", qPrintable(reply->errorString()), qPrintable(QString::number(reply->error())));

        QByteArray data = reply->readAll();
        delete reply;

        Mat m = imdecode(Mat(1, data.size(), CV_8UC1, data.data()), 1);
        if (m.data) t.append(m);

        return t;
    }

    void write(const Template &t) const
    {
        (void) t;
        qFatal("Not supported.");
    }
};

BR_REGISTER(Format, urlFormat)

} // namespace br

#include "format/url.moc"
