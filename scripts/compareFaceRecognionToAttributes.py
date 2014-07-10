import os 
import sys

if not (len(sys.argv) == 2 or len(sys.argv) == 4):
    print("ERROR: Input requires 1 or 3 input parameters\n Usage: compareFaceRecognitionToAttributes.py <path_to_data> [optional]<<path_to_query_parameter> <path_to_target_parameter>>")
else:
    data = sys.argv[1]
    compareData = "../data/MEDS/img"
    attrDir = "Attributes"
    attrPath = attrDir + "/all.model"
    mask = "MEDS.mask"
    if len(sys.argv) == 4:
        query = sys.argv[2]
        target = sys.argv[3]
    else:
        query = "../data/MEDS/sigset/MEDS_frontal_query.xml"
        target = "../data/MEDS/sigset/MEDS_frontal_target.xml"

    #Create Evaluation Mask
    if not os.path.isfile(mask):
        os.system("br -makeMask ../data/MEDS/sigset/MEDS_frontal_target.xml ../data/MEDS/sigset/MEDS_frontal_query.xml " + mask)

    #Train FaceRecognition Algorithm (Already trained from "make install")
    #os.system("br -algorithm FaceRecognition -path " + data + " -train results1194v2.turk faceRecognition.model")
    #Train Attributes Algorithm
    if not os.path.isfile(attrPath):
        os.system("mkdir -p " + attrDir)
        os.system("br -algorithm AllAttributesMatching -path " + data + " -train results1194v2.turk " + attrPath)

    #Run FaceRecognition Comparison
    os.system("br -path " + compareData + " -algorithm FaceRecognition -compare " + target + " " + query + " faceRecognition.mtx")
    #Run Attributes Comparison
    os.system("br -path " + compareData + " -TurkTargetHuman false -TurkQueryMachine true -algorithm " + attrPath + " -compare " + target + " " + query + " attributes.mtx")

    #Fuse the Matricies
    os.system("br -fuse faceRecognition.mtx attributes.mtx ZScore Sum fusion.mtx")

    #Evaluate all three matricies
    os.system("br -eval faceRecognition.mtx " + mask + " faceRecognition.csv")
    os.system("br -eval attributes.mtx " + mask + " attributes.csv")
    os.system("br -eval fusion.mtx " + mask + " faceRecognitionVSAttributes.csv")

    #Plot results
    os.system("br -plot faceRecognition.csv faceRecognition.pdf")
    os.system("br -plot attributes.csv attributes.pdf")
    os.system("br -plot faceRecognition.csv attributes.csv faceRecognitionVSAttributes.csv faceRecognitionVSAttributes.pdf")
