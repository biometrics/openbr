#!/bin/bash

echo '<?xml version="1.0" encoding="UTF-8"?>'
echo '<biometric-signature-set>'
for ((person=$1; person <= $2; person++)); do
	printf "\t<biometric-signature name=\"person%02d\">\n" $person
	for vidClass in {'boxing','handclapping','handwaving','jogging','running','walking'}; do
	
		for i in {1..4}; do
			# person13_handclapping_d3 is missing
			if [ $vidClass = 'handclapping' -a $person -eq 13 -a $i -eq 3 ]; then
				continue
			fi
			# person01_boxing_d4 was deleted (corrupted file)
			if [ $vidClass = 'boxing' -a $person -eq 1 -a $i -eq 4 ]; then
				continue
			fi
			
			printf "\t\t<presentation Label=\"$vidClass\" file-name=\"%s/person%02d_%s_d%d_uncomp.avi\" />\n" $vidClass $person $vidClass $i
		done
	done
	echo -e "\t</biometric-signature>"
done
echo '</biometric-signature-set>'
