import os
import sys

if not (len(sys.argv)) == 3:
    print("ERROR: Input requires 3 parameters\nUsage: attributesWeighted.py <min-weight-FR> <max-weight-FR>")
else:
    mask = "MEDS.mask"
    fr_matrix = "faceRecognition.mtx"
    attr_matrix = "attributes.mtx"
    min_weight = sys.argv[1]
    max_weight = sys.argv[2]
    constant = 1

    if not os.path.isfile(mask):
        print("ERROR: No mask found, run compareFaceRecognitionToAttributes.py first then try again")
    elif not os.path.isfile(fr_matrix):
        print("ERROR: No face recognition matrix found, run compareFaceRecognitionToAttributes.py first then try again")
    elif not os.path.isfile(attr_matrix):
        print("ERROR: No attributes matrix found, run compareFaceRecognitionToAttributes.py first then try again")
    else:
        for i in range(int(min_weight), int(max_weight)):
            print("Using weights " + str(i) + ":" + str(constant))
            os.system("br -fuse " + fr_matrix + " " + attr_matrix + " ZScore Sum" + str(i) + ":" + str(constant) + " weightedFusion.mtx")
            os.system("br -eval weightedFusion.mtx " + mask + " weightedFusion.csv")
            os.system("br -eval faceRecognition.mtx " + mask + " faceRecognition.csv")
            os.system("br -eval attributes.mtx " + mask + " attributes.csv")
            os.system("br -plot faceRecognition.csv attributes.csv weightedFusion.csv weightedFusion_" + str(i) + ".pdf")
