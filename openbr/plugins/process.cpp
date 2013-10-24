
#include <QBuffer>
#include <QLocalServer>
#include <QLocalSocket>
#include <QProcess>
#include <QUuid>

#include <iostream>
#include <fstream>

#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

enum SignalType
{
    INPUT_AVAILABLE,
    OUTPUT_AVAILABLE,
    SHOULD_END
};

class EnrollmentWorker
{
public:
    QLocalServer inbound;
    QLocalSocket outbound;
    QLocalSocket * receiver;

    ~EnrollmentWorker()
    {
        delete transform;
    }

    br::Transform * transform;

    void connections(const QString & baseName)
    {
        inbound.listen(baseName+"_worker");
        outbound.connectToServer(baseName+"_master");
        inbound.waitForNewConnection(-1);
        receiver =  inbound.nextPendingConnection();
        outbound.waitForConnected(-1);
    }

    void workerLoop()
    {
        SignalType signal;

        forever
        {
            while (receiver->bytesAvailable() < qint64(sizeof(signal))) {
                receiver->waitForReadyRead(-1);
            }
            receiver->read((char *) &signal, sizeof(signal));

            if (signal == SHOULD_END) {
                outbound.close();
                inbound.close();
                break;
            }

            qint64 inBufferSize;
            while (receiver->bytesAvailable() < qint64(sizeof(inBufferSize))) {
                receiver->waitForReadyRead(-1);
            }
            receiver->read((char *) &inBufferSize, sizeof(inBufferSize));

            QByteArray inArray(inBufferSize,'0');

            qint64 arrayPosition = 0;
            while (arrayPosition < inBufferSize) {
                if (!receiver->bytesAvailable())
                    receiver->waitForReadyRead(-1);
                arrayPosition += receiver->read(inArray.data()+arrayPosition, receiver->bytesAvailable());
            }

            TemplateList inList;
            TemplateList outList;
            // deserialize the template list
            QDataStream deserializer(inArray);
            deserializer >> inList;

            // and project it
            transform->projectUpdate(inList,outList);

            // serialize the output list
            QBuffer outBuff;
            outBuff.open(QBuffer::ReadWrite);
            QDataStream serializer(&outBuff);
            serializer << outList;

            // send the size of the buffer
            //qint64 bufferSize = outBuff.size();
            qint64 bufferSize = outBuff.data().size();
            outbound.write((char *) &bufferSize, sizeof(bufferSize));

            outbound.write(outBuff.data().data(), bufferSize);
            while (outbound.bytesToWrite() > 0) {
                outbound.waitForBytesWritten(-1);
            }
        }
    }


};

void WorkerProcess::mainLoop()
{
    processInterface = new EnrollmentWorker();
    processInterface->transform = Transform::make(this->transform,NULL);
    processInterface->connections(baseName);
    processInterface->workerLoop();
    delete processInterface;
}

/*!
 * \ingroup transforms
 * \brief Interface to a separate process
 * \author Charles Otto \cite caotto
 */
class ProcessWrapperTransform : public TimeVaryingTransform
{
    Q_OBJECT

    Q_PROPERTY(QString transform READ get_transform WRITE set_transform RESET reset_transform)
    BR_PROPERTY(QString, transform, "")

    QString baseKey;
    QProcess workerProcess;

    QLocalServer inbound;
    QLocalSocket outbound;
    QLocalSocket * receiver;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {
        if (!processActive)
        {
            activateProcess();
        }

        SignalType signal = INPUT_AVAILABLE;
        outbound.write((char *) &signal, sizeof(SignalType));

        QBuffer inBuffer;
        inBuffer.open(QBuffer::ReadWrite);
        QDataStream serializer(&inBuffer);
        serializer << src;

        qint64 in_size = inBuffer.size();
        outbound.write((char *) &in_size, sizeof(in_size));

        outbound.write(inBuffer.data(), in_size);

        while (outbound.bytesToWrite() > 0) {
            outbound.waitForBytesWritten(-1);
        }

        qint64 out_size;

        // read the size
        receiver->waitForReadyRead(-1);
        receiver->read((char *) &out_size, sizeof(out_size));
        QByteArray outBuffer(out_size,'0');

        // read the (serialized) output templatelist
        qint64 arrayPosition = 0;
        while (arrayPosition < out_size) {
            if (!receiver->bytesAvailable())
                receiver->waitForReadyRead(-1);
            arrayPosition += receiver->read(outBuffer.data()+arrayPosition, receiver->bytesAvailable());
        }
        // and deserialize it.
        QDataStream deserialize(outBuffer);
        deserialize >> dst;
    }


    void train(const TemplateList& data)
    {
        (void) data;
    }

    // create the process
    void init()
    {
        processActive = false;
    }

    void activateProcess()
    {
        processActive = true;

        // generate a uuid for our local servers
        QUuid id = QUuid::createUuid();
        baseKey = id.toString();

        QStringList argumentList;
        argumentList.append("-useGui");
        argumentList.append("0");
        argumentList.append("-algorithm");
        argumentList.append(transform);
        argumentList.append("-path");
        argumentList.append(Globals->path);
        argumentList.append("-parallelism");
        argumentList.append(QString::number(0));
        argumentList.append("-slave");
        argumentList.append(baseKey);

        // start listening
        inbound.listen(baseKey+"_master");

        workerProcess.setProcessChannelMode(QProcess::ForwardedChannels);
        workerProcess.start("br", argumentList);
        workerProcess.waitForStarted(-1);

        // blocking wait for the connection from the worker process
        inbound.waitForNewConnection(-1);
        receiver = inbound.nextPendingConnection();

        // Now, create our connection to the worker process.
        outbound.connectToServer(baseKey+"_worker");
        outbound.waitForConnected(-1);
    }

    bool timeVarying() const {
        return false;
    }

    ~ProcessWrapperTransform()
    {
        // end the process
        if (this->processActive) {

            SignalType signal = SHOULD_END;
            outbound.write((char *) &signal, sizeof(SignalType));
            outbound.waitForBytesWritten(-1);
            outbound.close();

            workerProcess.waitForFinished(-1);
            inbound.close();
            processActive = false;
        }
    }

public:
    bool processActive;
    ProcessWrapperTransform() : TimeVaryingTransform(false,false) { processActive = false; }
};

BR_REGISTER(Transform, ProcessWrapperTransform)


}


#include "process.moc"




