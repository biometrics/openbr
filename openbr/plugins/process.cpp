

#include <QBuffer>
#include <QCoreApplication>
#include <QLocalServer>
#include <QLocalSocket>
#include <QMutex>
#include <QProcess>
#include <QUuid>
#include <QWaitCondition>

#include "openbr_internal.h"
#include "openbr/core/opencvutils.h"

using namespace cv;

namespace br
{

class CommunicationManager : public QObject
{
    Q_OBJECT
public:
    CommunicationManager()
    {
        moveToThread(QCoreApplication::instance()->thread());
        server.moveToThread(QCoreApplication::instance()->thread() );
        outbound.moveToThread(QCoreApplication::instance()->thread() );

        // signals for our sever
        connect(&server, SIGNAL(newConnection()), this, SLOT(receivedConnection() ));
        connect(this, SIGNAL(pulseStartServer(QString)), this, SLOT(startServerInternal(QString)), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(pulseOutboundConnect(QString)), this, SLOT(startConnectInternal(QString) ), Qt::BlockingQueuedConnection);


        // internals, cause work to be done by the main thread because reasons.
        connect(this, SIGNAL(pulseSignal()), this, SLOT(sendSignalInternal()), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(pulseReadSignal() ), this, SLOT(readSignalInternal()), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(pulseReadSerialized() ), this, SLOT(readSerializedInternal()), Qt::BlockingQueuedConnection); 
        connect(this, SIGNAL(pulseSendSerialized() ), this, SLOT(sendSerializedInternal() ), Qt::BlockingQueuedConnection);
        connect(this, SIGNAL(pulseShutdown() ), this, SLOT(shutdownInternal() ), Qt::BlockingQueuedConnection);

        // signals for our outbound connection
        connect(&outbound, SIGNAL(connected() ), this, SLOT(outboundConnected() ));
        connect(&outbound, SIGNAL(disconnected() ), this, SLOT(outboundDisconnected() ));

        connect(&outbound, SIGNAL(error(QLocalSocket::LocalSocketError)), this, SLOT(outboundConnectionError(QLocalSocket::LocalSocketError) ) );
        connect(&outbound, SIGNAL(stateChanged(QLocalSocket::LocalSocketState)), this, SLOT(outboundStateChanged(QLocalSocket::LocalSocketState) ) );

        inbound = NULL;
    }

    enum SignalType
    {
        INPUT_AVAILABLE,
        OUTPUT_AVAILABLE,
        SHOULD_END
    };


public slots:
    // matching server signals
    void receivedConnection()
    {
        inbound = server.nextPendingConnection();
        connect(inbound, SIGNAL(disconnected() ), this, SLOT(inboundDisconnected() ));
        connect(inbound, SIGNAL(error(QLocalSocket::LocalSocketError)), this, SLOT(inboundConnectionError(QLocalSocket::LocalSocketError) ) );
        connect(inbound, SIGNAL(stateChanged(QLocalSocket::LocalSocketState)), this, SLOT(inboundStateChanged(QLocalSocket::LocalSocketState) ) );

        receivedWait.wakeAll();
    }

    // matching outbound socket signals

    // Oh boy.
    void outboundConnected()
    {
        outboundWait.wakeAll();
    }

    // Oh no!
    void outboundDisconnected()
    {
        //qDebug() << key << " outbound socket has disconnected";
    }

    // informative.
    void outboundConnectionError(QLocalSocket::LocalSocketError socketError)
    {
        (void) socketError;
        //qDebug() << key << " outbound socket error " << socketError;
    }

    void outboundStateChanged(QLocalSocket::LocalSocketState socketState)
    {
        (void) socketState;
        //qDebug() << key << " outbound socket state changed to " << socketState;
    }

    // matching inbound socket signals
    void inboundDisconnected()
    {
        //qDebug() << key << " inbound socket has disconnected";
    }

    void inboundConnectionError(QLocalSocket::LocalSocketError socketError)
    {
        (void) socketError;
        //qDebug() << key << " inbound socket error " << socketError;
    }

    void inboundStateChanged(QLocalSocket::LocalSocketState socketState)
    {
        (void) socketState;
        //qDebug() << key << " inbound socket state changed to " << socketState;
    }

    void startServerInternal(QString serverName)
    {
         if (!server.isListening()) {
            bool listen_res = server.listen(serverName);
            if (!listen_res)
                qDebug() << key << " Failed to start server at " << serverName;
         }
    }

    void sendSignalInternal()
    {
        SignalType signal=  sendType;
        qint64 signal_wrote = outbound.write((char *) &signal, sizeof(signal));
        if (signal_wrote != sizeof(signal))
            qDebug() << key << " inconsistent signal size";

        bool res = outbound.waitForBytesWritten(-1);
        if (!res)
            qDebug() << key << " failed to wait for bytes written in signal size";
    }

    void readSignalInternal()
    {
        while (inbound->bytesAvailable() < qint64(sizeof(readSignal)) ) {
            bool size_ready = inbound->waitForReadyRead(-1);
            if (!size_ready)
            {
                qDebug("Failed to received object size in signal!");
            }
        }

        qint64 signalBytesRead = inbound->read((char *) &readSignal, sizeof(readSignal));
        if (signalBytesRead != sizeof(readSignal))
            qDebug("Inconsistent signal size read!");

        if (readSignal == SHOULD_END)
        {
            server.close();
            outbound.abort();
            inbound->abort();
        }
    }

    void sendSerializedInternal()
    {
        qint64 serializedSize = writeArray.size();
        qint64 size_wrote = outbound.write((char *) &serializedSize, sizeof(serializedSize));
        if (size_wrote != sizeof(serializedSize)) {
            qDebug() << key << "inconsistent size sent in send data!";
            return;
        }
        bool res = outbound.waitForBytesWritten(-1);
        
        if (!res) {
            qDebug() << key << " wait for bytes failed!";
            return;
        }

        qint64 data_wrote = outbound.write(writeArray.data(), serializedSize);
        if (data_wrote != serializedSize)
            qDebug() << key << " inconsistent data written!";

        while (outbound.bytesToWrite() > 0) {
            bool write_res = outbound.waitForBytesWritten(-1);
            if (!write_res) {
                qDebug() << key << " wait for bytes failed!";
                return;
            }
        }

        return;

    }

    void readSerializedInternal()
    {
        qint64 bufferSize;
        while (inbound->bytesAvailable() < qint64(sizeof(bufferSize))) {
            bool size_ready = inbound->waitForReadyRead(-1);
            if (!size_ready)
            {
                qDebug() << key << " Failed to received object size in read data!";
                qDebug() << key << "inbound status: " << inbound->state() << " error: " << inbound->errorString();
                return;
            }
        }
        qint64 sizeBytesRead = inbound->read((char *) &bufferSize, sizeof(bufferSize));
        if (sizeBytesRead != sizeof(bufferSize)) {
            qDebug("failed to read size of buffer!");
            return;
        }

        // allocate the input buffer
        readArray.resize(bufferSize);

        // read the data, we may get it in serveral bursts
        qint64 arrayPosition = 0;
        while (arrayPosition < bufferSize) {
            if (!inbound->bytesAvailable()) {
                bool ready_res = inbound->waitForReadyRead(-1);

                if (!ready_res) {
                    qDebug() << key << "failed to wait for data!";
                    return;
                }
            }

            // how many bytes do we still need?
            qint64 bytes_remaining = bufferSize - arrayPosition;

            if (bytes_remaining < inbound->bytesAvailable() )
            {
                qDebug() << key << "!!!excessive bytes received";
            }
            arrayPosition += inbound->read(readArray.data()+arrayPosition, qMin(inbound->bytesAvailable(), bytes_remaining));
        }
        if (arrayPosition != bufferSize)
            qDebug() << key <<  "Read wrong size object!";

    }

    void shutdownInternal()
    {
        outbound.abort();
        inbound->abort();
        server.close();
    }


    void startConnectInternal(QString remoteName)
    {
        outbound.connectToServer(remoteName);
    }


signals:
    void pulseStartServer(QString serverName);
    void pulseSignal();
    void pulseReadSignal();
    void pulseReadSerialized();
    void pulseSendSerialized();
    void pulseShutdown();
    void pulseOutboundConnect(QString serverName);


public:
    QByteArray readArray;
    QByteArray writeArray;

    SignalType readSignal;
    QMutex receivedLock;
    QWaitCondition receivedWait;

    QMutex outboundLock;
    QWaitCondition outboundWait;

    QString key;
    QString serverName;
    QString remoteName;

    QLocalSocket * inbound;
    QLocalSocket outbound;
    QLocalServer server;


    void waitForInbound()
    {
        QMutexLocker locker(&receivedLock);
        while (!inbound || inbound->state() != QLocalSocket::ConnectedState) {
            bool res = receivedWait.wait(&receivedLock,30*1000);
            if (!res)
            {
                qDebug() << key << " " << QThread::currentThread() << " waiting timed out, server thread is " << server.thread() << " application thread " << QCoreApplication::instance()->thread();
            }
        }
    }

    void connectToRemote(const QString & remoteName)
    {
        emit pulseOutboundConnect(remoteName);

        QMutexLocker locker(&outboundLock);
        while (outbound.state() != QLocalSocket::ConnectedState) {
            outboundWait.wait(&outboundLock,30*1000);

        }
    }

    SignalType getSignal()
    {
        emit pulseReadSignal();
        return readSignal;
    }


    template<typename T>
    bool readData(T & input)
    {
        emit pulseReadSerialized();
        QDataStream deserializer(readArray);
        deserializer >> input;
        return true;
    }

    template<typename T>
    bool sendData(const T & output)
    {
        QBuffer buffer;
        buffer.open(QBuffer::ReadWrite);

        QDataStream serializer(&buffer);
        serializer << output;
        writeArray = buffer.data();
        emit pulseSendSerialized();
        return true;
    }

    SignalType sendType;
    void sendSignal(SignalType signal)
    {
        sendType = signal;
        if (QThread::currentThread() == this->thread() )
            this->sendSignalInternal();
        else
            emit pulseSignal();
    }

    void startServer(QString server)
    {
        emit pulseStartServer(server);
    }

    void shutdown()
    {
        emit pulseShutdown();
    }

 
};

class EnrollmentWorker : public QObject
{
    Q_OBJECT
public:
    CommunicationManager * comm;
    QString name;

    ~EnrollmentWorker()
    {
        delete transform;
        delete comm;
    }

    br::Transform * transform;

public:
    void connections(const QString & baseName)
    {
        comm = new CommunicationManager();
        name = baseName;
        comm->key = "worker_"+baseName.mid(1,5);
        comm->startServer(baseName+"_worker");
        comm->connectToRemote(baseName+"_master");

        comm->waitForInbound();
    }

    void workerLoop()
    {
        QString sign = "worker " + name;
        CommunicationManager::SignalType signal;
        forever
        {
            signal= comm->getSignal();

            if (signal == CommunicationManager::SHOULD_END) {
                break;
            }
            TemplateList inList;
            TemplateList outList;

            comm->readData(inList);
            transform->projectUpdate(inList,outList);
            comm->sendData(outList);
        }
        comm->shutdown();
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

void shutUp(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    // Qt you have no idea how much I don't care about you.
    // Please tell me more about how you want every single god damn thing to be created from and used by exactly one thread.
    // It does not matter, so shut up already.
    // p.s. I hope you die.
    (void) type; (void) context; (void) msg;
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

    QProcess * workerProcess;
    CommunicationManager * comm;

    void projectUpdate(const TemplateList &src, TemplateList &dst)
    {

        if (!processActive)
        {
            activateProcess();
        }
        comm->sendSignal(CommunicationManager::INPUT_AVAILABLE);

        comm->sendData(src);

        comm->readData(dst);
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
        comm = new CommunicationManager();
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

        comm->key = "master_"+baseKey.mid(1,5);

        comm->startServer(baseKey+"_master");

        workerProcess = new QProcess();
        workerProcess->setProcessChannelMode(QProcess::ForwardedChannels);
        workerProcess->start("br", argumentList);
        workerProcess->waitForStarted(-1);

        comm->waitForInbound();
        comm->connectToRemote(baseKey+"_worker");
    }

    bool timeVarying() const {
        return false;
    }

    ~ProcessWrapperTransform()
    {
        // end the process
        if (this->processActive) {
            comm->sendSignal(CommunicationManager::SHOULD_END);

            // I don't even want to talk about it.
            qInstallMessageHandler(shutUp);
            workerProcess->waitForFinished(-1);
            delete workerProcess;
            qInstallMessageHandler(0);

            processActive = false;
            comm->inbound->abort();
            comm->outbound.abort();
            comm->server.close();
            delete comm;
        }
    }

public:
    bool processActive;
    ProcessWrapperTransform() : TimeVaryingTransform(false,false) { processActive = false; }
};

BR_REGISTER(Transform, ProcessWrapperTransform)

}

#include "process.moc"
