

"""
Exercice 2 : MATRIX MULTIPLICATION WITH PARALLELIZATION USING BLOCS AND GRIDS

-> Remark 1: Code adapted from: https://shephexd.github.io/development/2017/02/19/pycuda.html

-> Remark 2:  If using google colab:
    * Click on Runtime (excecution) and select Change runtime type (modifier le type d'excecution).
    * Then select GPU in Hardware Acceleration (accélérateur matériel)
    * Start your session by installing pycuda with the command: \" !pip install pycuda \"

"""

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#1) INITIALISATION
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

import numpy as np
import pycuda
from pycuda import driver, compiler, gpuarray, tools
import time

# -- initialize the device
import pycuda.autoinit


#get device information
MyDevice=pycuda.driver.Device(0)
attributes=MyDevice.get_attributes()


# define the (square) matrix size
MATRIX_SIZE = 1024

# create two random square matrices
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)


#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#2) MATRIX MULTIPLICATION ON THE CPU
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

# compute reference on the CPU to verify GPU computation
time_start=time.time()
c_cpu = np.dot(a_cpu, b_cpu)
time_end=time.time()
cpt_time_cpu=time_end-time_start
print('enlapsed time (CPU):',cpt_time_cpu,' seconds')


#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#3) MAKE AND COMPILE THE KERNEL FOR MATRIX MULTIPLICATION ON THE GPU
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

#define the kernel
kernel_code_templates = """
__global__ void MatrixMulKernel_one_row_per_thread(float *a, float *b, float *c)
{
    int tx = threadIdx.x;
    float c_value;

      for (int j = 0; j < %(MATRIX_SIZE)s; ++j) {
        c_value = 0.;
        for (int k = 0; k < %(MATRIX_SIZE)s; ++k) {
          c_value += a[tx * %(MATRIX_SIZE)s + k] * b[k * %(MATRIX_SIZE)s + j];
        }
        c[tx * %(MATRIX_SIZE)s + j]=c_value;
      }
}
"""



#Remark that a single block is used here and that it contains MATRIX_SIZE threads, each
#of them treating a row of C. We check below whether our GPU (ot more specifically
#our device) will be able to multiply the matrices A and B using kernel we just
#defined.

MTPB=MyDevice.__getattr__('MAX_THREADS_PER_BLOCK')
if MTPB <= MATRIX_SIZE:
  print("The device is OK to multiply A and B using MatrixMulKernel_one_row_per_thread ")
else:
  print("The matrices A and B are too large to be multiplied using MatrixMulKernel_one_row_per_thread")



# get the kernel code from the template, compile the kernel, and get the
# kernel function from the compiled module
kernel_codes = kernel_code_templates % {'MATRIX_SIZE': MATRIX_SIZE}
mod = compiler.SourceModule(kernel_codes)
matrixmul_one_row_per_bloc = mod.get_function("MatrixMulKernel_one_row_per_thread")

#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
#4) MATRIX MULTIPLICATION ON THE GPU
#+ + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +

# call the kernel on the card -- sequential matrix multiplication
time_start=time.time()
matrixmul_one_row_per_bloc(a_gpu , b_gpu , c_gpu , block = (MATRIX_SIZE, 1, 1))
time_end=time.time()
cpt_time_gpu=time_end-time_start

print('enlapsed time (GPU - one_row_per_bloc):',cpt_time_gpu,' seconds')

print('The computations were '+str(np.round(cpt_time_cpu/cpt_time_gpu,2))+' times faster by using the GPU instead of the CPU')

print('norm(c_cpu-c_gpu)='+str(np.linalg.norm(c_cpu - c_gpu.get())))





"""
QUESTION 1 : comprenez et executez le code. Le gain entre les temps de calculs
             sur le GPU et les CPU devraient maintenant tourner largement en
             faveur du GPU.

QUESTION 2 : Pourriez-vous réduire un peu plus les temps de calculs en utilisant
             la mémoire __shared__ ? Les temps d'acces en lecture et en
             ecriture sont beaucoup plus rapide pour cette mémoire que pour
             la mémoire __global__ . Cependant :
               -> Elle est partagée entre tous les threads d'un bloc de
                  calculs. Il faut alors faire attention a ce que les
                  threads n'ecrasent pas les données '__shared__' utilisées
                  dans d'autres threads
               -> La taille de cette mémoire est limitée. On peut récupérer
                  sa taille en octets en utilisant
                     MyDevice.__getattr__('MAX_SHARED_MEMORY_PER_BLOCK')
                  sachant que par exemple un float32 fait 4 octets
               -> Cette information n'est visible qu'a l'interieur d'un bloc.

Pour information :
  -> un vecteur de taille 10 peut par exemple etre déclaré dans la mémoire
     __shared__ comme ceci :

              __shared__ float my_shared_vec[10];

  -> une sous partie de la matrice 'a' peut par exemple etre transférée dans
     ce vecteur partagé comme ça :

            for (k = 0; k < 10; ++k)
              my_shared_vec[k]=a[tx * %(MATRIX_SIZE)s + (20+k)];

      Il faudra cependant faire attention à ce que tous les threads (chacun
      étant lié à un tx) ne copient pas des informations différentes dans
      la même case mémoire !!!

"""



MSMPB=MyDevice.__getattr__('MAX_SHARED_MEMORY_PER_BLOCK');
max_floats_per_block=MSMPB/4   #4 is the float size in Bites
print('Maximum number of floats that can be allocated in a block:',max_floats_per_block)

MTPB=MyDevice.__getattr__('MAX_THREADS_PER_BLOCK')
print('Maximum number of threads (ie block_size_x * block_size_y * block_size_z) available in a block:',MTPB)
