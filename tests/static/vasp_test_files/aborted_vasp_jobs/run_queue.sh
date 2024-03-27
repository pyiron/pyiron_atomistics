#!/bin/bash
#SBATCH --partition=p.cmfe
#SBATCH --ntasks=120
#SBATCH --constraint='[swi1|swi1|swi2|swi3|swi4|swi5|swi6|swi7|swi8|swi9|swe1|swe2|swe3|swe4|swe5|swe6|swe7]'
#SBATCH --time=5760
#SBATCH --output=time.out
#SBATCH --error=error.out
#SBATCH --job-name=pi_20331947
#SBATCH --chdir=/cmmc/u/aabdelkawy/ahmed/vdm_project/5.TiN111_Ni111_4_7_23/TiN111_Ni111_interface/vasp_jobs/defective_interface/tin_ni_interf_surf_ni_sub_Nb_hdf5/tin_ni_interf_surf_ni_sub_Nb
#SBATCH --get-user-env=L

pwd; 
echo Hostname: `hostname`
echo Date: `date`
echo JobID: $SLURM_JOB_ID

python -m pyiron_base.cli wrapper -p /cmmc/u/aabdelkawy/ahmed/vdm_project/5.TiN111_Ni111_4_7_23/TiN111_Ni111_interface/vasp_jobs/defective_interface/tin_ni_interf_surf_ni_sub_Nb_hdf5/tin_ni_interf_surf_ni_sub_Nb -j 20331947