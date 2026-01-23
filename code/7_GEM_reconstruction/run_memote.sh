# Bash script for running memote on all models in folder
echo "Folder with gapfilled models: $1"

for file in $1/*.xml ; do
    echo "Run memote on $file"
    name=$(basename -- "$file")
    new_fn=$1/"${name%%.*}"_memote_report.html
    memote report snapshot --filename $new_fn $file --skip test_detect_energy_generating_cycles --skip test_find_reactions_unbounded_flux_default_condition
    echo "Saved $new_fn"
done