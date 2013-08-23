#!/bin/bash

# prints out a <presentation> element
# additional arguments after the first two are rectangle coordinates
# from that, it will create a nested <data:bbox> element (from the ViPER standard)
# http://viper-toolkit.sourceforge.net/docs/file/
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
# export writePresentation so xargs can call it using bash -c below
export -f writePresentation
SEDREGEX='s/.*(\([0-9]*\), \([0-9]*\)) - (\([0-9]*\), \([0-9]*\))/writeIt \1 \2 \3 \4/'

echo '<?xml version="1.0" encoding="UTF-8"?>'
echo '<biometric-signature-set>'

# print out the positive image sigs
for fullpath in INRIAPerson/$1/pos/*; do
    # get just the filename, minus the path
    filename=$(basename "$fullpath")
    echo -e "\t<biometric-signature name=\"${filename%.*}\">"

    # if this folder has annotations, add bounding boxes
    if [ -d INRIAPerson/$1/annotations ]; then
        annotation="INRIAPerson/$1/annotations/${filename%.*}.txt"
        grep 'Bounding box' $annotation | sed "$SEDREGEX" | xargs -n 5 bash -c "writePresentation $1 $filename \$@"
    # otherwise, just the normal presentation
    else
        writePresentation $1 $filename
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
