#!/bin/bash

############################################################
# Help                                                     #
############################################################
Help()
{
   # Display Help
   echo "This script estimates the normal vectors for the"
   echo "input Point Cloud, via CloudCompare using quadric"
   echo "fitting for a radius of 5."
   echo
   echo "Syntax: estimateNormals <pc_name.ply>"
   echo
}

############################################################
# Main program                                             #
############################################################

if [[ $# -eq 0 ]] ; then
    Help
    exit 1
fi


inputPC=${1}


if [ -f "$inputPC" ]; then
    if [[ $inputPC == *.ply ]]; then
        outputPC=${inputPC:: -4}_n.ply
        
        cloudcompare.CloudCompare -SILENT -O $inputPC -C_EXPORT_FMT 'PLY' -AUTO_SAVE OFF -OCTREE_NORMALS 5 -MODEL QUADRIC -SAVE_CLOUDS FILE $outputPC
    else 
        echo "$inputPC is not a PLY file."
    fi
else 
    echo "File $inputPC does not exist."
fi

