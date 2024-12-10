python run.py -c=configs/sim_qrvae.yaml
python run.py -c=configs/sim_vae.yaml
python evaluate.py -c=configs/sim_qrvae.yaml
python evaluate.py -c=configs/sim_vae.yaml


python run.py -c=configs/sim_c_cvae.yaml
python run.py -c=configs/sim_c_vae.yaml
python run.py -c=configs/sim_c_cqrvae.yaml
python run.py -c=configs/sim_c_qrvae.yaml
# python evaluate.py -c=configs/sim_c_cvae.yaml
# python evaluate.py -c=configs/sim_c_vae.yaml
# python evaluate.py -c=configs/sim_c_cqrvae.yaml
# python evaluate.py -c=configs/sim_c_qrvae.yaml

python evaluate_cond.py -c=configs/sim_c_cvae.yaml
python evaluate_cond.py -c=configs/sim_c_vae.yaml
python evaluate_cond.py -c=configs/sim_c_cqrvae.yaml
python evaluate_cond.py -c=configs/sim_c_qrvae.yaml

python run.py -c=configs/sim_c_cqrvae.yaml
python evaluate_cond.py -c=configs/sim_c_cqrvae.yaml

# python evaluate.py -c=configs/sim_c_cqrvae.yaml
# python evaluate.py -c=configs/sim_c_qrvae.yaml

