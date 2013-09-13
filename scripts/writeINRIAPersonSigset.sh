#!/bin/bash

# prints out a <data:bbox> element from rectangle coordinates
# (from the ViPER standard: http://viper-toolkit.sourceforge.net/docs/file/)
printBBox()
{
    width=$(($3-$1))
    height=$(($4-$2))
    echo -e "\t\t\t<data:bbox height=\"$height\" width=\"$width\" x=\"$1\" y=\"$2\" />"
}
# export printBBox so xargs can call it using bash -c below
export -f printBBox
SEDREGEX='s/.*(\([0-9]*\), \([0-9]*\)) - (\([0-9]*\), \([0-9]*\))/printBBox \1 \2 \3 \4/'

echo '<?xml version="1.0" encoding="UTF-8"?>'
echo '<biometric-signature-set>'

# print out the positive image sigs
for fullpath in INRIAPerson/$1/pos/*; do
    # get just the filename, minus the path
    filename=$(basename "$fullpath")
    echo -e "\t<biometric-signature name=\"${filename%.*}\">"

    # if this folder has annotations, add bounding boxes
    echo -en "\t\t<presentation Label=\"pos\" file-name=\"$1/pos/$filename\""
    if [ -d INRIAPerson/$1/annotations ]; then
        echo ">"
        annotation="INRIAPerson/$1/annotations/${filename%.*}.txt"
        grep 'Bounding box' $annotation | sed "$SEDREGEX" | xargs -n 5 bash -c 'printBBox $@'
        echo -e "\t\t</presentation>"
    # otherwise, just end the presentation
    else
        echo " />"
    fi

    echo -e '\t</biometric-signature>'
done

# print out the negative image sigs
for fullpath in INRIAPerson/$1/neg/*; do
    filename=$(basename "$fullpath")
    echo -e "\t<biometric-signature name=\"${filename%.*}\">"
    echo -e "\t\t<presentation Label=\"neg\" file-name=\"$1/neg/$filename\" />"
    echo -e '\t</biometric-signature>'
done

echo '</biometric-signature-set>'
