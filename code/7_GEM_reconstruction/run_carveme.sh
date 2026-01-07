# Bash script for making not gap-filled carveme models
echo "Input folder: $1"
echo "output folder: $2"

for file in $1/*.faa ; do
    echo "Create carveme model from $file"
    name=$(basename -- "$file")
    new_fn=$2/"${name%%.*}".xml
    carve $file -o $new_fn --solver gurobi --universe-file ../../data/7_GEM_reconstruction/carveme_universe_bacteria_fixed.xml
done