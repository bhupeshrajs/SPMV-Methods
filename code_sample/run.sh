#!/bin/bash


echo "********************     cant      ************************"

./spmv -mat ../../Matrices/cant_62451.mtx -ivec ../../Vectors/vector_cant_62451.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/cant_62451.mtx -ivec ../../Vectors/vector_cant_62451.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/cant_62451.mtx -ivec ../../Vectors/vector_cant_62451.txt -alg design -blockSize $1 -blockNum $2

echo "******************** circuit5M_dc ************************"

./spmv -mat ../../Matrices/circuit5M_dc_3523317.mtx -ivec ../../Vectors/vector_circuit5M_dc_3523317.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/circuit5M_dc_3523317.mtx -ivec ../../Vectors/vector_circuit5M_dc_3523317.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/circuit5M_dc_3523317.mtx -ivec ../../Vectors/vector_circuit5M_dc_3523317.txt -alg design -blockSize $1 -blockNum $2


echo "******************** consph ************************"

./spmv -mat ../../Matrices/consph_83334.mtx -ivec ../../Vectors/vector_consph_83334.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/consph_83334.mtx -ivec ../../Vectors/vector_consph_83334.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/consph_83334.mtx -ivec ../../Vectors/vector_consph_83334.txt -alg design -blockSize $1 -blockNum $2


echo "******************** FullChip ************************"

./spmv -mat ../../Matrices/FullChip_2987012.mtx -ivec ../../Vectors/vector_FullChip_2987012.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/FullChip_2987012.mtx -ivec ../../Vectors/vector_FullChip_2987012.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/FullChip_2987012.mtx -ivec ../../Vectors/vector_FullChip_2987012.txt -alg design -blockSize $1 -blockNum $2


echo "******************** mac_econ_fwd500 ************************"

./spmv -mat ../../Matrices/mac_econ_fwd500_206500.mtx -ivec ../../Vectors/vector_mac_econ_fwd500_206500.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/mac_econ_fwd500_206500.mtx -ivec ../../Vectors/vector_mac_econ_fwd500_206500.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/mac_econ_fwd500_206500.mtx -ivec ../../Vectors/vector_mac_econ_fwd500_206500.txt -alg design -blockSize $1 -blockNum $2


echo "******************** mc2depi ************************"

./spmv -mat ../../Matrices/mc2depi_525825.mtx -ivec ../../Vectors/vector_mc2depi_525825.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/mc2depi_525825.mtx -ivec ../../Vectors/vector_mc2depi_525825.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/mc2depi_525825.mtx -ivec ../../Vectors/vector_mc2depi_525825.txt -alg design -blockSize $1 -blockNum $2


echo "******************** pdb1HYS ************************"

./spmv -mat ../../Matrices/pdb1HYS_36417.mtx -ivec ../../Vectors/vector_pdb1HYS_36417.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/pdb1HYS_36417.mtx -ivec ../../Vectors/vector_pdb1HYS_36417.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/pdb1HYS_36417.mtx -ivec ../../Vectors/vector_pdb1HYS_36417.txt -alg design -blockSize $1 -blockNum $2


echo "******************** pwtk ************************"

./spmv -mat ../../Matrices/pwtk_217918.mtx -ivec ../../Vectors/vector_pwtk_217918.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/pwtk_217918.mtx -ivec ../../Vectors/vector_pwtk_217918.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/pwtk_217918.mtx -ivec ../../Vectors/vector_pwtk_217918.txt -alg design -blockSize $1 -blockNum $2


echo "******************** rail4284 ************************"

./spmv -mat ../../Matrices/rail4284_1096894.mtx -ivec ../../Vectors/vector_rail4284_1096894.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/rail4284_1096894.mtx -ivec ../../Vectors/vector_rail4284_1096894.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/rail4284_1096894.mtx -ivec ../../Vectors/vector_rail4284_1096894.txt -alg design -blockSize $1 -blockNum $2


echo "******************** scircuit ************************"

./spmv -mat ../../Matrices/scircuit_170998.mtx -ivec ../../Vectors/vector_scircuit_170998.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/scircuit_170998.mtx -ivec ../../Vectors/vector_scircuit_170998.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/scircuit_170998.mtx -ivec ../../Vectors/vector_scircuit_170998.txt -alg design -blockSize $1 -blockNum $2


echo "******************** shipsec1 ************************"

./spmv -mat ../../Matrices/shipsec1_140874.mtx -ivec ../../Vectors/vector_shipsec1_140874.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/shipsec1_140874.mtx -ivec ../../Vectors/vector_shipsec1_140874.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/shipsec1_140874.mtx -ivec ../../Vectors/vector_shipsec1_140874.txt -alg design -blockSize $1 -blockNum $2


echo "******************** turon_m ************************"

./spmv -mat ../../Matrices/turon_m_189924.mtx -ivec ../../Vectors/vector_turon_m_189924.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/turon_m_189924.mtx -ivec ../../Vectors/vector_turon_m_189924.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/turon_m_189924.mtx -ivec ../../Vectors/vector_turon_m_189924.txt -alg design -blockSize $1 -blockNum $2


echo "******************** watson2 ************************"


./spmv -mat ../../Matrices/watson_2_677224.mtx -ivec ../../Vectors/vector_watson_2_677224.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/watson_2_677224.mtx -ivec ../../Vectors/vector_watson_2_677224.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/watson_2_677224.mtx -ivec ../../Vectors/vector_watson_2_677224.txt -alg design -blockSize $1 -blockNum $2


echo "******************** webbase-1M ************************"

./spmv -mat ../../Matrices/webbase-1M_1000005.mtx -ivec ../../Vectors/vector_webbase_1M_1000005.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/webbase-1M_1000005.mtx -ivec ../../Vectors/vector_webbase_1M_1000005.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/webbase-1M_1000005.mtx -ivec ../../Vectors/vector_webbase_1M_1000005.txt -alg design -blockSize $1 -blockNum $2


echo "******************** rma10 ************************"

./spmv -mat ../../Matrices/rma10_64835.mtx -ivec ../../Vectors/vector_rma10_46835.txt -alg atomic -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/rma10_64835.mtx -ivec ../../Vectors/vector_rma10_46835.txt -alg segment -blockSize $1 -blockNum $2
./spmv -mat ../../Matrices/rma10_64835.mtx -ivec ../../Vectors/vector_rma10_46835.txt -alg design -blockSize $1 -blockNum $2

