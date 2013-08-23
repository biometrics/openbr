#!/bin/bash

writePresentation()
{
    pres="\t\t<presentation Label=\"pos\" file-name=\"$1/pos/$2\""
    if [ "$#" -eq 6 ]; then
        pres="$pres>"
        width=$(($5-$3))
        height=$(($6-$4))
        pres="$pres\n\t\t\t<data:bbox height=\"$height\" width=\"$width\" x=\"$3\" y=\"$4\" />"
        pres="$pres\n\t\t</presentation>"
    else
        pres="$pres />"
    fi
    printf "$pres\n"
}
# for xargs calling bash below
export -f writePresentation
SEDREGEX='s/.*(\([0-9]*\), \([0-9]*\)) - (\([0-9]*\), \([0-9]*\))/writeIt \1 \2 \3 \4/'

echo '<?xml version="1.0" encoding="UTF-8"?>'
echo '<biometric-signature-set>'
for fullpath in INRIAPerson/$1/pos/*; do
    filename=$(basename "$fullpath")
    echo -e "\t<biometric-signature name=\"${filename%.*}\">"
    if [ -d INRIAPerson/$1/annotations ]; then
        annotation="INRIAPerson/$1/annotations/${filename%.*}.txt"
        grep 'Bounding box' $annotation | sed "$SEDREGEX" | xargs -n 5 bash -c "writePresentation $1 $filename \$@"
    else
        writePresentation $1 $filename
    fi
    echo -e '\t</biometric-signature>'
done
for fullpath in INRIAPerson/$1/neg/*; do
    filename=$(basename "$fullpath")
    echo -e "\t<biometric-signature name=\"${filename%.*}\">"
    echo -e "\t\t<presentation Label=\"neg\" file-name=\"$1/neg/$filename\" />"
    echo -e '\t</biometric-signature>'
done
echo '</biometric-signature-set>'
