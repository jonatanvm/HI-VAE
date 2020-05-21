#source ~/venv/bin/activate

declare dataset="hospital"
declare batch_size=2000

declare epochs=150
declare m_perc=33
declare mask=1

declare data_train_file=${dataset}/data_train.csv
declare data_test_file=${dataset}/data_test.csv
declare types_file=${dataset}/data_types.csv
declare miss_file_train=${dataset}/Missing33_${mask}.csv
declare miss_file_test=${dataset}/Missing100_${mask}.csv
declare true_miss_file=${dataset}/MissingTrue.csv


declare model="model_HIVAE_inputDropout"
declare z_dim=30
declare y_dim=10
declare s_dim=20



train_model(){
  python hospital/scripts.py train
  python main_scripts.py --model_name $1 --batch_size ${batch_size} --epochs ${epochs} \
  --data_file ${data_train_file} --types_file ${types_file} --miss_file ${miss_file_train} \
  --dim_latent_z $2 --dim_latent_y $3 --dim_latent_s $4 \
  --save_file ${save_file} \
  #--true_miss_file ${true_miss_file}
}

test_model(){
  python hospital/scripts.py test
  python main_scripts.py --model_name $1 --batch_size 10000000 --epochs 1 \
  --data_file ${data_test_file} --types_file ${types_file} --miss_file ${miss_file_test} \
  --dim_latent_z $2 --dim_latent_y $3 --dim_latent_s $4 \
  --save_file ${save_file} --train 0 --restore 1 \
  #--true_miss_file ${true_miss_file}
}


declare save_file=${model}_kaggle_${dataset}_Missing${m_perc}_${mask}_z${z_dim}_y${y_dim}_s${s_dim}_batch${batch_size}_epochs${epochs}_2_4_7_8_9
train_model ${model} ${z_dim} ${y_dim} ${s_dim}
test_model ${model} ${z_dim} ${y_dim} ${s_dim}
