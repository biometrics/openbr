0.4.0 - ??/??/??
================
* Added -evalLandmarking and -plotLandmarking for evaluating and plotting landmarking accuracy (#9)
* Added -evalDetection and -plotDetection for evaluating and plotting object detection accuracy (#9)
* Deprecated Transform::backProject

0.3.0 - 5/22/13
===============
* Added wrapper to NEC Latent SDK
* Enrolling files/folders are now sorted naturally instead of alpha numerically
* YouTubeFacesDBTransform implements Dr. Wolf's experimental protocol
* NEC3 refactored
* Updated transform API to add support for time-varying transforms per issue (#23)
* Refactored File class to improve point and rect storage (#22)
* Added algorithm to show face detection results (#25)
* Reorganized GUI code and include paths (#31)
* 'br -daemon' to listen for commands on stdin
* Generalized 'br -convert', now requires three parameters
* Official icon, thanks @sklum!

0.2.0 - 2/23/13
===============
* FaceRecognition new distance metric
  - 0 to 1 range indicating match probability
* Qt 4.8 -> Qt 5.0
* Cleaner plots generated with 'br -plot'
* Stasm and FLandmark wrappers
* Improved demographic filtering speed
  - br::Context::demographicFilters -> br::Context::filters
  - MetadataDistance -> FilterDistance
* PipeDistance
* ImpostorUniquenessMeasureTransform
* MatchProbabilityDistance
* CrossValidation framework
  - br::Context::crossValidate
  - CrossValidationTransform
  - CrossValidationDistance

0.1.0 - 1/27/13
===============
First official release!
