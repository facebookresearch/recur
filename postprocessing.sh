# Post-processing
# assemble raw data file from prefixes
cat */data.prefix \
| awk 'BEGIN{PROCINFO["sorted_in"]="@val_num_desc"}{c[$0]++}END{for (i in c) printf("%i|%s\n",c[i],i)}' \
> equations

# remove duplicates
uniq equations > equations_unique
mv equations_unique equations

# create train, valid and test samples
python ~/recur/split_data.py equations 10000
mv equations equations.train
