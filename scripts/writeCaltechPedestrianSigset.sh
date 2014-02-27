#!/bin/bash

echo '<?xml version="1.0" encoding="UTF-8"?>'
echo '<biometric-signature-set>'
for ((set=$1; set <= $2; set++)); do
	echo -e "\t<biometric-signature name=\"\">"
	for vid in `printf "set%02d/*.seq" $set`; do
		if [ $3 == "train" ]; then
			vbb=`echo "vbb=\"annotations/$vid\"" | sed s/seq/vbb/`
		fi
		printf "\t\t<presentation file-name=\"vid/$vid\" $vbb />\n"
	done
	echo -e "\t</biometric-signature>"
done
echo '</biometric-signature-set>'
