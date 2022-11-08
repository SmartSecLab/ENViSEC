# shortening commands in this Makefile

# local dirs/files to exclude from rsync (also for occurences in subdirectories)
excludes=.pio .idea .dvc .git .neptune .dvcignore .gitattributes .gitignore venv results data .DS_Store __pycache__ 
includes=data/classes.npy data/predict/conn.log
local=ex3
dnat=ex3-local
ex3project=ENViSEC/ENViSEC-dev
data_path=../../General/dataset/iot_23_datasets_small/chunks
model_path=results/dnn-100-base
raspi2=guru@rasp2
jetson=guru@jetson

all: help

help:
	@echo "The Makefile shortens some frequent commands"
	@echo "Such as:"
	@echo "make deploy     # increamental deployment of the current project to eX3"
	@echo "make plots      # fetch plots from ex3 to local"
	@echo "make venv       # create a venv (only works on eX3)"

deploy:
	rsync -avzhHP -e 'ssh' $(addprefix --exclude , $(patsubst %,'%',$(excludes))) . ${dnat}:${ex3project}
	
data:
	rsync -avzhHP -e 'ssh' $(addprefix --exclude , $(patsubst %,'%',$(excludes))) ${data_path} ${dnat}:${ex3project}/data/raw/chunks/

deploy-auto:
	@case `curl ifconfig.me` in \
    158.36.4.* ) case `hostname -f` in \
                 *.cm.cluster) echo "looks like we're on eX3, this command is meant to run from your local machine..." ;; \
                 *) rsync -avzhHP -e 'ssh' $(addprefix --exclude , $(patsubst %,'%',$(excludes))) . ${local}:${ex3project} ;; \
                 esac ;; \
	*) rsync -avzhHP -e 'ssh -p 60441' $(addprefix --exclude , $(patsubst %,'%',$(excludes))) . ${dnat}:${ex3project} ;; \
	esac

plots:
	@case `curl ifconfig.me` in \
    158.36.4.* ) case `hostname -f` in \
                 *.cm.cluster) echo "looks like we're on eX3, this command is meant to run from your local machine..." ;; \
                 *) rsync -avzhHP -e 'ssh' ${local}:${ex3project}/results --exclude "**/tb" --include "*/" --include "*.jpg" --exclude "*" . && find results -empty -type d -delete ;; \
                 esac ;; \
	*) rsync -avzhHP -e 'ssh -p 60441' ${dnat}:${ex3project}/results --exclude "**/tb" --include "*/" --include "*.jpg" --exclude "*" . && find results -empty -type d -delete ;; \
	esac
	
venv:
	@case `hostname -f` in \
	*.cm.cluster)  (  eval "$$(grep '^module ' slurm_train.sh)"; \
	                  python3 -m venv venv; \
	                  source venv/bin/activate; \
	                  pip install -r requirements.txt; \
	               ) ;; \
	*) 	echo "looks like we're not on eX3, this command is not meant to run from your local machine..." ;; \
	esac

cast:
	# rsync -avzhHP $(addprefix --exclude , $(patsubst %,'%',$(excludes))) $(addprefix --include , $(patsubst %,'%',$(includes))) . ${raspi2}:ENViSEC
	rsync -avzhHP $(addprefix --exclude , $(patsubst %,'%',$(excludes))) $(addprefix --include , $(patsubst %,'%',$(includes))) . ${jetson}:ENViSEC

cast-model:
	rsync -avzhHP ${model_path} ${raspi2}:ENViSEC/results
	rsync -avzhHP data/classes.npy ${raspi2}:ENViSEC/data
	rsync -avzhHP ${model_path} ${jetson}:ENViSEC/results
	rsync -avzhHP data/classes.npy ${jetson}:ENViSEC/data
