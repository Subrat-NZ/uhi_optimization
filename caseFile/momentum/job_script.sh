#!/bin/bash
#PBS -l select=1:ncpus=2:mpiprocs=2
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o pbs_openfoam.log
#PBS -V

cd "$PBS_O_WORKDIR" || exit 1

echo "=============================================="
echo "PBS job started"
echo "Job ID      : $PBS_JOBID"
echo "Host        : $(hostname)"
echo "Work dir    : $PBS_O_WORKDIR"
echo "Start time  : $(date)"
echo "=============================================="

# Load OpenFOAM
. /opt/openfoam8/etc/bashrc

# Recommended OpenMPI settings for PBS/OpenFOAM
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_mapping_policy=slot

# Optional: limit OpenMP threads so each MPI rank does not oversubscribe
export OMP_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/of_Allrun" ./of_Allrun

chmod +x of_Allrun
./of_Allrun

echo "=============================================="
echo "PBS job finished"
echo "End time: $(date)"
echo "=============================================="