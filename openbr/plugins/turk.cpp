#include "openbr_internal.h"
#include "openbr/core/common.h"
#include "openbr/core/opencvutils.h"
#include "openbr/core/qtutils.h"

namespace br
{

/*!
 * \ingroup galleries
 * \brief For Amazon Mechanical Turk datasets
 * \author Scott Klum \cite sklum
 */
class turkGallery : public Gallery
{
    Q_OBJECT

    struct Attribute : public QStringList
    {
        QString name;
        Attribute(const QString &str = QString())
        {
            const int i = str.indexOf('[');
            name = str.mid(0, i);
            if (i != -1)
                append(str.mid(i+1, str.length()-i-2).split(","));
        }

        Attribute normalized() const
        {
            bool ok;
            QList<float> values;
            foreach (const QString &value, *this) {
                values.append(value.toFloat(&ok));
                if (!ok)
                    qFatal("Can't normalize non-numeric vector!");
            }

            Attribute normal(name);
            float sum = Common::Sum(values);
            if (sum == 0) sum = 1;
            for (int i=0; i<values.size(); i++)
                normal.append(QString::number(values[i] / sum));
            return normal;
        }
    };

    TemplateList readBlock(bool *done)
    {
        *done = true;
        QStringList lines = QtUtils::readLines(file);
        QList<Attribute> headers;
        if (!lines.isEmpty())
            foreach (const QString &header, parse(lines.takeFirst()))
                headers.append(header);

        TemplateList templates;
        foreach (const QString &line, lines) {
            QStringList words = parse(line);
            if (words.size() != headers.size())
                qFatal("turkGallery invalid column count");

            File f;
            f.name = words[0];
            f.set("Label", words[0].mid(0,5));

            for (int i=1; i<words.size(); i++) {
                Attribute ratings = Attribute(words[i]).normalized();
                if (headers[i].size() != ratings.size())
                    qFatal("turkGallery invalid attribute count");
                for (int j=0; j<ratings.size(); j++)
                    f.set(headers[i].name + "_" + headers[i][j], ratings[j]);
            }
            templates.append(f);
        }

        return templates;
    }

    void write(const Template &)
    {
        qFatal("turkGallery write not implemented.");
    }
};

BR_REGISTER(Gallery, turkGallery)

/*!
 * \ingroup transforms
 * \brief Convenience class for training turk attribute regressors
 * \author Josh Klontz \cite jklontz
 */
class TurkClassifierTransform : public Transform
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key STORED false)
    Q_PROPERTY(QStringList values READ get_values WRITE set_values RESET reset_values STORED false)
    Q_PROPERTY(bool isMeta READ get_isMeta WRITE set_isMeta RESET reset_isMeta STORED false)
    BR_PROPERTY(QString, key, QString())
    BR_PROPERTY(QStringList, values, QStringList())
    BR_PROPERTY(bool, isMeta, false)

    Transform *child;

    void init()
    {
        QStringList classifiers;
        isMeta = false; // Trying to satisfy a bug. This is not used anyways.        
        foreach (const QString &value, values) 
            classifiers.append(QString("(SVM(RBF,EPS_SVR,returnDFVal=true,inputVariable=%1,outputVariable=predicted_%1)%2)").arg(key + "_" + value, isMeta ? QString("+Average+SaveMat(predicted_%1)").arg(value) : QString()));
        child = Transform::make(classifiers.join("/") + (classifiers.size() > 1 ? "+Cat" : ""));
    }

    void train(const QList<TemplateList> &data)
    {
        child->train(data);
    }

    void project(const Template &src, Template &dst) const
    {
        child->project(src, dst);
    }

    void store(QDataStream &stream) const
    {
        child->store(stream);
    }

    void load(QDataStream &stream)
    {
        child->load(stream);
    }
};

BR_REGISTER(Transform, TurkClassifierTransform)

/*!
 * \ingroup distances
 * \brief Unmaps Turk HITs to be compared against query mats
 * \author Scott Klum \cite sklum
 */
class TurkDistance : public Distance
{
    Q_OBJECT
    Q_PROPERTY(QString key READ get_key WRITE set_key RESET reset_key)
    Q_PROPERTY(QStringList values READ get_values WRITE set_values RESET reset_values STORED false)
    BR_PROPERTY(QString, key, QString())
    BR_PROPERTY(QStringList, values, QStringList())

    bool targetHuman;
    bool queryMachine;

    void init()
    {
        targetHuman = Globals->property("TurkTargetHuman").toBool();
        queryMachine = Globals->property("TurkQueryMachine").toBool();
    }

    cv::Mat getValues(const Template &t) const
    {
        QList<float> result;
        foreach (const QString &value, values)
            result.append(t.file.get<float>(key + "_" + value));
        return OpenCVUtils::toMat(result, 1);
    }

    float compare(const Template &target, const Template &query) const
    {
        const cv::Mat a = targetHuman ? getValues(target) : target.m();
        const cv::Mat b = queryMachine ? query.m() : getValues(query);
        return -norm(a, b, cv::NORM_L1);
    }
};

BR_REGISTER(Distance, TurkDistance)

/*!
 * \ingroup initializers
 * \brief Initializes global abbreviations with implemented algorithms for attributes
 * \author Babatunde Ogunfemi \cite baba1472
 */
class AttributeAlgorithmsInitializer : public Initializer
{
    Q_OBJECT

    void initialize() const
    {
        // Constants
        QString BASE="Open+PP5Register+Rename(PP5_Landmark0_Right_Eye,Affine_0)+Rename(PP5_Landmark1_Left_Eye,Affine_1)+Affine(192,240,.345,.475)+Cvt(Gray)+Stasm(false,true,[(66.24,114),(125.76,114)])";
        QString SUBSPACE ="Blur(1.1)+Gamma(0.2)+DoG(1,2)+ContrastEq(0.1,10)+LBP(1,2)+RectRegions(8,8,4,4)+Hist(59)+Cat+PCA(0.95)";

        QString NOSE="RectFromStasmNoseWithBridge+ROI+Resize(36,24)+" + SUBSPACE;
        QString MOUTH="RectFromStasmMouth+ROI+Resize(24,36)+" + SUBSPACE;
        QString EYES="RectFromStasmEyes+ROI+Resize(24,36)+" + SUBSPACE;
        QString HAIR="RectFromStasmHair+ROI+Resize(24,36)+" + SUBSPACE;
        QString BROW="RectFromStasmBrow+ROI+Resize(24,36)+" + SUBSPACE;
        QString JAW="RectFromStasmJaw+ROI+Resize(36,36)+" + SUBSPACE;
        QString FACE = "Crop(24,30,144,190)+Resize(36,36)+" +  SUBSPACE;

        // All Attributes
        Globals->abbreviations.insert("AllAttributes", "AttributeBrow/AttributeMouth/AttributeEyes/AttributeFace/AttributeHair/AttributeNose/AttributeJaw");
        Globals->abbreviations.insert("AllAttributesMatching", "(AttributeBrow)/(AttributeMouth)/(AttributeEyes)/(AttributeFace)/(AttributeHair)/(AttributeNose)/(AttributeJaw):AttributeMatch");

        //Individual Attributes
        Globals->abbreviations.insert("AttributeBrow", "(" + BASE+ "+" + BROW + "+"
        "TurkClassifier(eyebrowposition,[closebrows,highbrows],3)/"
        "TurkClassifier(unibrow,[unibrow],3)/"
        "TurkClassifier(eyebroworientation,[eyebrowsdown,eyebrowsuptodown],3)/"
        "TurkClassifier(thickeyebrows,[thickeyebrows,lighteyebrows],3))");
        Globals->abbreviations.insert("AttributeMouth", "(" + BASE + "+" + MOUTH + "+"
        "TurkClassifier(smiling,[smiling],3)/"
        "TurkClassifier(lipthickness,[cherry,big,small],3)/"
        "TurkClassifier(mouthbite,[underbite,overbite],3)/"
        "TurkClassifier(mouthopen,[closed,noteeth,halfteeth,allteeth],3)/"
        "TurkClassifier(mouthwidth,[small,wide],3)/"
        "TurkClassifier(mustache,[nomustache,linemustache,lightmustache,normalmustache,down],3)/"
        "TurkClassifier(mouthasymmetry,[asymmetrical],3))");
        Globals->abbreviations.insert("AttributeEyes", "(" + BASE + "+" + EYES + "+ "
        "TurkClassifier(eyeseparation,[close,wide],3)/"
        "TurkClassifier(eyeslant,[slant2,slant1,wild],3)/"
        "TurkClassifier(benteyes,[bent])/"
        "TurkClassifier(eyecolor,[darkeyes,lighteyes],3)/"
        "TurkClassifier(baggyeyes,[baggy],3)/" 
        "TurkClassifier(almondeyes,[almond],3)/"
        "TurkClassifier(buriedeyes,[buriedeyes],3)/"
        "TurkClassifier(sleepyeyes,[sleepy],3)/"
        "TurkClassifier(lineeyes,[line],3)/"
        "TurkClassifier(roundeyes,[round],3)/"
        "TurkClassifier(sharpeyes,[sharp],3)/"
        "TurkClassifier(smalleyes,[smalleyes],3)/"
        "TurkClassifier(glasses,[glasses],3)/"
        "TurkClassifier(eyelashvisibility,[feweyelashes],3))");
        Globals->abbreviations.insert("AttributeFace", "(" + BASE + "+" + FACE + "+"
        "TurkClassifier(gender,[male],3)/"
        "TurkClassifier(faceshape,[round,triangular,rectangular],3)/"
        "TurkClassifier(cheekdensity,[puffy,in,normal],3)/"
        "TurkClassifier(facemarks,[scars,moles,normal],3)/"
        "TurkClassifier(facelength,[long],3)/"
        "TurkClassifier(nosetoeyedist,[short,long],3)/"
        "TurkClassifier(nosetomouthdist,[long,small],3))");
        Globals->abbreviations.insert("AttributeHair", "(" + BASE + "+" + HAIR + "+"
        "TurkClassifier(foreheadwrinkles,[wrinkled],3)/"
        "TurkClassifier(foreheadsize,[smallforehead,largeforehead],3)/"
        "TurkClassifier(haircolor,[darkhair,lighthair,greyhair],3)/"
        "TurkClassifier(hairdensity,[thick,bald,thin,halfbald],3)/"
        "TurkClassifier(widowspeak,[widowspeak],3)/"
        "TurkClassifier(hairstyle,[curlyhair],3))");
        Globals->abbreviations.insert("AttributeNose", "(" + BASE + "+" + NOSE + "+"
        "TurkClassifier(noseorientation,[upnose,downnose],3)/"
        "TurkClassifier(nosewidth,[small,thick],3)/"
        "TurkClassifier(nosesize,[smallnose,bignose],3)/"
        "TurkClassifier(brokennose,[broken],3))");
        Globals->abbreviations.insert("AttributeJaw", "(" + BASE + "+" + JAW + "+"
        "TurkClassifier(beard,[nobeard,bigbeard,lightbeard,goatee,linebeard,normalbeard,lincolnbeard],3)/"
        "TurkClassifier(chinsize,[shortchin,longchin],3))");
        Globals->abbreviations.insert("AttributeMatch", "Fuse(["
        "Turk(eyebrowposition,[closebrows,highbrows],3)," 
        "Turk(unibrow,[unibrow],3)," 
        "Turk(eyebroworientation,[eyebrowsdown,eyebrowsuptodown],3)," 
        "Turk(thickeyebrows,[thickeyebrows,lighteyebrows],3)," 
        "Turk(smiling,[smiling],3)," 
        "Turk(lipthickness,[cherry,big,small],3)," 
        "Turk(mouthbite,[underbite,overbite],3)," 
        "Turk(mouthopen,[closed,noteeth,halfteeth,allteeth],3)," 
        "Turk(mouthwidth,[small,wide],3)," 
        "Turk(mustache,[nomustache,linemustache,lightmustache,normalmustache,down],3)," 
        "Turk(mouthasymmetry,[asymmetrical],3)," 
        "Turk(eyeseparation,[close,wide],3)," 
        "Turk(eyeslant,[slant2,slant1,wild],3)," 
        "Turk(benteyes,[bent],3)," 
        "Turk(eyecolor,[darkeyes,lighteyes],3)," 
        "Turk(baggyeyes,[baggy],3)," 
        "Turk(almondeyes,[almond],3)," 
        "Turk(buriedeyes,[buriedeyes],3)," 
        "Turk(sleepyeyes,[sleepy],3)," 
        "Turk(lineeyes,[line],3)," 
        "Turk(roundeyes,[round],3)," 
        "Turk(sharpeyes,[sharp],3)," 
        "Turk(smalleyes,[smalleyes],3)," 
        "Turk(glasses,[glasses],3)," 
        "Turk(eyelashvisibility,[feweyelashes],3)," 
        "Turk(gender,[male],3)," 
        "Turk(faceshape,[round,triangular,rectangular],3)," 
        "Turk(cheekdensity,[puffy,in,normal],3)," 
        "Turk(facemarks,[scars,moles,normal],3)," 
        "Turk(facelength,[long],3)," 
        "Turk(nosetoeyedist,[short,long],3)," 
        "Turk(nosetomouthdist,[long,small],3)," 
        "Turk(foreheadwrinkles,[wrinkled],3)," 
        "Turk(foreheadsize,[smallforehead,largeforehead],3)," 
        "Turk(haircolor,[darkhair,lighthair,greyhair],3)," 
        "Turk(hairdensity,[thick,bald,thin,halfbald],3)," 
        "Turk(widowspeak,[widowspeak],3)," 
        "Turk(hairstyle,[curlyhair],3)," 
        "Turk(noseorientation,[upnose,downnose],3)," 
        "Turk(nosewidth,[small,thick],3)," 
        "Turk(nosesize,[smallnose,bignose],3)," 
        "Turk(brokennose,[broken],3)," 
        "Turk(beard,[nobeard,bigbeard,lightbeard,goatee,linebeard,normalbeard,lincolnbeard],3)," 
        "Turk(chinsize,[shortchin,longchin],3)])");
    }
};

BR_REGISTER(Initializer, AttributeAlgorithmsInitializer)

} // namespace br

#include "turk.moc"
