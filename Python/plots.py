# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:09:00 2025

@author: korbi
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import grksvd
import svdviaqr
import numpy as np

#Relative time and energy cost constants

add = 1
mul = 1
sqr = 15
div = 20

gadd = 4
gmul = 4

chip_conf = 10000
chip_mul = 50

qr_iter_multiplier = 13.88
grk_iter_multiplier = 1.47

cpu_energy = 375
gpu_energy = 32.24

cconf_energy = 640
cmul_energy = 320

#Time Estimation

def getClassicBidiagTime(n):
    addp = 4/3 * n * n * n * n
    addp += 2/3 * n* n * n
    addp += 17/3 * n * n
    addp += 1/3 * n
    addp += -8
    
    mulp = 4/3 * n * n * n * n
    mulp += 4/3 * n* n * n
    mulp += 11/3 * n * n
    mulp += -4/3 * n
    mulp += -5
    
    divp = 2*n - 2
    sqrp = 4*n - 4
    
    return (add *addp + mul * mulp + div * divp + sqr * sqrp)

def getClassicQRTime(n):
    iterations = grk_iter_multiplier * n
    
    addp = 4*n*n + 15*n - 16
    mulp = 8*n*n + 20*n - 27
    divp = 4*n-3
    sqrp = 2*n-1
    
    time = 0;
    time += addp * add
    time += mulp * mul
    time += divp * div
    time += sqrp * sqr
    
    return time * iterations

def getClassicalBidiagTimeUsingGPU(n):
    addp = 3*n*n+2*n-5
    mulp = 3*n*n-3
    gaddp = 2*n*n+6*n-8
    gmulp = 2*n*n+4*n-6
    divp = 2*n-2
    sqrp = 4*n-4
    
    return (add *addp + gadd * gaddp + mul * mulp + gmul * gmulp + div * divp + sqr * sqrp)


def getClassicalQRTimeUsingGPU(n):
    iterations = grk_iter_multiplier * n
    
    addp = 7*n
    mulp = 4*n+5
    gaddp = 4*n-4
    gmulp = 8*n-8
    divp = 4*n-3
    sqrp = 2*n-1
    
    time = 0;
    time += addp * add
    time += mulp * mul
    time += divp * div
    time += sqrp * sqr
    time += gaddp * gadd
    time += gmulp * gmul
    
    return time * iterations

def getGRKBidiagPhotonicTime(n):
    addp = 5*n*n - 10*n + 5
    mulp = 4*n*n - 8*n + 4
    divp = 3*n*n - 6*n + 3
    sqrp = 2*n*n - 4*n + 2
    cmp = 4*n*n
    ccp = 2*n
    
    return (add *addp + mul * mulp + div * divp + sqr * sqrp + cmp * chip_mul + ccp * chip_conf)
    
    
def getGRKQRPhotonicTime(n):
    iterations = grk_iter_multiplier * n
    
    addp = 18*n - 16
    mulp = 28*n - 27
    divp = 4*n - 3
    sqrp = 2*n - 1
    
    cconfp = 2
    cmultp = 2 * n
    
    time = 0;
    time += addp * add
    time += mulp * mul
    time += divp * div
    time += sqrp * sqr
    
    time += cconfp * chip_conf
    time += cmultp * chip_mul
    
    return time * iterations
    

def getALTQRPhotonicTime(n):
    iterations = qr_iter_multiplier * n
    
    addp = 5*n*(n-1)
    mulp = 4*n*(n-1)
    divp = 3*n*(n-1)
    sqrp = 2*n*(n-1)
    
    cconfp = 2 * n
    cmultp = 4 * n * n
    
    time = 0;
    time += addp * add
    time += mulp * mul
    time += divp * div
    time += sqrp * sqr
    
    time += cconfp * chip_conf
    time += cmultp * chip_mul
    
    return time * iterations

def getALTQRGPUTime(n):
    iterations = qr_iter_multiplier * n
    
    addp = 3*n*n+5*n-4
    mulp = 3*n*n+3*n-3
    gaddp = 2*n*n+10*n-5
    gmulp = 2*n*n+8*n-4
    divp = 2*n-1
    sqrp = 4*n-2
    
    time = 0;
    time += addp * add
    time += mulp * mul
    time += divp * div
    time += sqrp * sqr
    time += gaddp * gadd
    time += gmulp * gmul
    
    return time * iterations
    
def getALTQRLinTime(n):
    iterations = qr_iter_multiplier * n
    
    addp = 4/3*n*n*n*n+14/3*n*n*n+20/3*n*n+16/3*n-7
    mulp = 4/3*n*n*n*n+16/3*n*n*n+17/3*n*n+5/3*n-5
    divp = 2*n-1
    sqrp = 4*n-2
    
    time = 0;
    time += addp * add
    time += mulp * mul
    time += divp * div
    time += sqrp * sqr
    
    return time * iterations 
















#Energy Estimation



def getALTQRLinEnergy(n):
    t = getALTQRLinTime(n)
    return t * cpu_energy

def getALTQRGPUEnergy(n):
    iterations = qr_iter_multiplier * n
    
    #Gpu Operations (actual GPU count)
    gaddp = 4/3*n*n*n*n+14/3*n*n*n+11/3*n*n+1/3*n-3
    gmulp = 4/3*n*n*n*n+16/3*n*n*n+8/3*n*n-4/3*n-2
    
    addp = 3*n*n+5*n-4
    mulp = 3*n*n+3*n-3
    divp = 2*n-1
    sqrp = 4*n-2
    
    time = 0;
    time += addp * add
    time += mulp * mul
    time += divp * div
    time += sqrp * sqr
    
    e = time * cpu_energy
    e += (gaddp + gmulp) * gpu_energy
    
    return e * iterations

def getALTQRPhotonicEnergy(n):
    iterations = qr_iter_multiplier * n
    
    addp = 5*n*(n-1)
    mulp = 4*n*(n-1)
    divp = 3*n*(n-1)
    sqrp = 2*n*(n-1)
    
    cconfp = 2 * n
    cmultp = 4 * n * n
    
    time = 0;
    time += addp * add
    time += mulp * mul
    time += divp * div
    time += sqrp * sqr
    
    energy = time * cpu_energy
    
    energy += cconfp * cconf_energy * n * (n-1)
    energy += cmultp * cmul_energy * n
    
    return energy * iterations

def getGRKLinEnergy(n):
    time = getClassicBidiagTime(n)
    time += getClassicQRTime(n)
    
    return time * cpu_energy

def getGRKGPUEnergy(n):
    addp = 3*n*n+2*n-5
    mulp = 3*n*n-3
    divp = 2*n-2
    sqrp = 4*n-4
    
    addp *= add
    mulp *= mul
    divp *= div
    sqrp *= sqr
    
    total_cpu_op_time = addp + mulp + divp + sqrp
    
    iterations = grk_iter_multiplier * n
    
    qraddp = 7*n
    qrmulp = 4*n+5
    qrdivp = 4*n-3
    qrsqrp = 2*n-1
    
    qraddp *= add
    qrmulp *= mul
    qrdivp *= div
    qrsqrp *= sqr
    
    total = qraddp + qrmulp + qrdivp + qrsqrp
    total_cpu_op_time += total * iterations
    total_cpu_energy_cost = total_cpu_op_time * cpu_energy
    
    #Gpu Operations (actual GPU count)
    gaddc1 = 4/3*n*n*n*n+2/3*n*n*n+8/3*n*n-5/3*n-3
    gmulc1 = 4/3*n*n*n*n+4/3*n*n*n+2/3*n*n-4/3*n-2
    
    gaddc2 = 4*n*n+8*n-16
    gmulc2 = 8*n*n+16*n-32
    
    total_gpu_ops = gaddc1 + gmulc1
    total_gpu_ops += (gaddc2 + gmulc2) * iterations
    total_gpu_energy_cost = total_gpu_ops * gpu_energy
    
    return total_cpu_energy_cost + total_gpu_energy_cost

def getGRKPhotonicEnergy(n):
    addp = 5*n*n - 10*n + 5
    mulp = 4*n*n - 8*n + 4
    divp = 3*n*n - 6*n + 3
    sqrp = 2*n*n - 4*n + 2
    
    t = add *addp + mul * mulp + div * divp + sqr * sqrp
    
    iterations = grk_iter_multiplier * n
    
    qraddp = 18*n-16
    qrmulp = 28*n-27
    qrdivp = 4*n-3
    qrsqrp = 2*n-1
    
    
    time = 0;
    time += qraddp * add
    time += qrmulp * mul
    time += qrdivp * div
    time += qrsqrp * sqr
    
    total_cpu_time = t + time * iterations
    total_cpu_energy_cost = total_cpu_time * cpu_energy
    
    cmp = 4*n*n
    ccp = 2*n
    
    cconfp = 2
    cmultp = 2 * n
    
    total_chip_confs = ccp + cconfp * iterations
    total_chip_ops = cmp + cmultp * iterations
    
    energy = total_chip_confs * cconf_energy * n * (n-1)
    energy += total_chip_ops * cmul_energy * n
    
    return total_cpu_energy_cost + energy
    
    



#Plot creation




def plot1():
    xValues = range(10, 1101)
    qrp = []
    qrlin = []
    rgkpho = []
    rgklin = []
    
    for i in xValues:
        qrp.append(0.25 * getALTQRPhotonicTime(i))
        qrlin.append(0.25 * getALTQRLinTime(i))
        rgkpho.append(0.25 * (getGRKBidiagPhotonicTime(i) + getGRKQRPhotonicTime(i)))
        rgklin.append(0.25 * (getClassicBidiagTime(i) + getClassicQRTime(i)))
    
    plt.plot(xValues, qrp, color='black', linestyle='--', label='QR-SVD H')
    plt.plot(xValues, qrlin, color='blue', linestyle='--', label='QR-SVD D-SC')
    plt.plot(xValues, rgkpho, color='black', linestyle='-', label='GRK-SVD H')
    plt.plot(xValues, rgklin, color='blue', linestyle='-', label='GRK-SVD D-SC')
    
    plt.legend()
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Estimated Runtime [s]')
    plt.title('Estimated Runtime Comparison')
    
    plt.xscale('log')
    plt.yscale('log')
   
    plt.gca().yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, pos: f"{v*1e-9:g}")
    )
  
    plt.savefig("plot1loglog.pdf", bbox_inches="tight")
    plt.close('all')
    
def plot2():
    xValues = range(10, 1101)

    qrp = []
    qrpar = []
    rgkpho = []
    rgkpar = []
    
    for i in xValues:
        qrp.append(0.25 * getALTQRPhotonicTime(i))
        qrpar.append(0.25 * (getALTQRGPUTime(i)))
        rgkpho.append(0.25 * (getGRKBidiagPhotonicTime(i) + getGRKQRPhotonicTime(i)))
        rgkpar.append(0.25 * (getClassicalBidiagTimeUsingGPU(i) + getClassicalQRTimeUsingGPU(i))) 
    
    plt.plot(xValues, qrp, color='black', linestyle='--', label='QR-SVD H')
    plt.plot(xValues, qrpar, color='purple', linestyle='--', label='QR-SVD D-MC')
    plt.plot(xValues, rgkpho, color='black', linestyle='-', label='GRK-SVD H')
    plt.plot(xValues, rgkpar, color='purple', linestyle='-', label='GRK-SVD D-MC')
    
    plt.legend()
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Estimated Runtime [s]')
    plt.title('Estimated Runtime Comparison')
     
    plt.xscale('log')
    plt.yscale('log')
        
    plt.gca().yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, pos: f"{v*1e-9:g}")
    )
        
    plt.savefig("plot2loglog.pdf", bbox_inches="tight")
    plt.close('all')


def plot3():
    xValues = range(10, 1101)

    qrp = []
    qrlin = []
    rgkpho = []
    rgklin = []
    
    for i in xValues:
        qrp.append(getALTQRPhotonicEnergy(i))
        qrlin.append(getALTQRLinEnergy(i))
        rgkpho.append(getGRKPhotonicEnergy(i))
        rgklin.append(getGRKLinEnergy(i))
    
    plt.plot(xValues, qrp, color='black', linestyle='--', label='QR-SVD H')
    plt.plot(xValues, qrlin, color='blue', linestyle='--', label='QR-SVD D-SC')
    plt.plot(xValues, rgkpho, color='black', linestyle='-', label='GRK-SVD H')
    plt.plot(xValues, rgklin, color='blue', linestyle='-', label='GRK-SVD D-SC')
    
    plt.legend()
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Estimated Energy Cost [J]')
    plt.title('Estimated Energy Consumption Comparison')
      
    plt.xscale('log')
    plt.yscale('log')

    plt.gca().yaxis.set_major_formatter(
           mticker.FuncFormatter(lambda v, pos: f"{v*1e-12:g}")
    )
        
    plt.savefig("plot3loglog.pdf", bbox_inches="tight")
    plt.close('all')
    
def plot4():
    xValues = range(10, 1101)
    qrp = []
    qrpar = []
    rgkpho = []
    rgkpar = []
    
    for i in xValues:
        qrp.append(getALTQRPhotonicEnergy(i))
        qrpar.append(getALTQRGPUEnergy(i))
        rgkpho.append(getGRKPhotonicEnergy(i))
        rgkpar.append(getGRKGPUEnergy(i))
        
    plt.plot(xValues, qrp, color='black', linestyle='--', label='QR-SVD H')
    plt.plot(xValues, qrpar, color='purple', linestyle='--', label='QR-SVD D-MC')
    plt.plot(xValues, rgkpho, color='black', linestyle='-', label='GRK-SVD H')
    plt.plot(xValues, rgkpar, color='purple', linestyle='-', label='GRK-SVD D-MC')
    
    plt.legend()
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Estimated Energy Cost [J]')
    plt.title('Estimated Energy Consumption Comparison')
     
    plt.xscale('log')
    plt.yscale('log')
        
    plt.gca().yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, pos: f"{v*1e-12:g}")
    )
    
    plt.savefig("plot4loglog.pdf", bbox_inches="tight")
    plt.close('all')


def plotGRKSVDconvergence(size,minIterations, maxIterations):
    matrix = []
    for i in range(200):
        if i % 20 == 0:
            print(i)
        matrix = np.random.rand(size, size)
        U, iterations, errors = grksvd.svd(matrix)
        errors.append(1e-6)
        
        for j in range(len(errors), maxIterations):
            errors.append(np.nan)
        
        errors = errors[minIterations:maxIterations]
        
        plt.plot(range(minIterations, maxIterations), errors, color='black', alpha=0.05)
    
    print(2)
    
    plt.title('Error Convergence for the GRK-SVD Algorithm')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.tight_layout()
    
    plt.savefig("plotERGRKSVD1.pdf", bbox_inches="tight")
    
    plt.show()
    return

def plotQRSVDconvergence(size,minIterations, maxIterations):
    matrix = []
    for i in range(200):
        matrix = np.random.rand(size, size)
        U, D, V, iterations, errors = svdviaqr.naive_svd(matrix)
        
        for j in range(len(errors), maxIterations):
            errors.append(np.nan)
        
        errors = errors[minIterations:maxIterations]
        
        plt.plot(range(minIterations, maxIterations), errors, color='black', alpha=0.05)
    
    plt.title('Error Convergence for the QR-SVD Algorithm')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.tight_layout()
    
    plt.savefig("plotERCQItQR.pdf", bbox_inches="tight")
    
    plt.show()
    return

def plot_required_iterations_QRSVD():
    xs = []
    ys = []

    with open("AltQRData.txt", "r") as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines

            values = list(map(int, line.split()))
            x = values[0]
            y_values = values[1:]

            xs.extend([x] * len(y_values))
            ys.extend(y_values)

    plt.figure()
    plt.scatter(xs, ys, alpha=0.05, color='blue')
    
    m = qr_iter_multiplier

    # create points for the line
    y_fit = m * np.arange(min(xs), max(xs)+1)
    
    
    plt.plot(range(min(xs), max(xs)+1), y_fit, color='black', label='13.88n')
    
    plt.title('Required Iterations to reach error of 10e-6')
    plt.xlabel('Dimension')
    plt.ylabel('Iterations')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("plotAltQRC.pdf", bbox_inches="tight")
    
    plt.show()
    
    
def plot_required_iterations_GRKSVD():
    xs = []
    ys = []

    with open("GRKData.txt", "r") as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines

            values = list(map(int, line.split()))
            x = values[0]
            y_values = values[1:]

            xs.extend([x] * len(y_values))
            ys.extend(y_values)

    plt.figure()
    plt.scatter(xs, ys, alpha=0.05, color='blue')
    
    m = grk_iter_multiplier

    # create points for the line
    y_fit = m * np.arange(min(xs), max(xs)+1)
    
    
    plt.plot(range(min(xs), max(xs)+1), y_fit, color='black', label='1.47n')
    
    plt.title('Required Iterations to reach error of 10e-6')
    plt.xlabel('Dimension')
    plt.ylabel('Iterations')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("plotGRKC.pdf", bbox_inches="tight")
    
    plt.show()
    
def timecheck_GRKSVD():
    xs = []
    ys = []

    with open("timeCheckGRK.txt", "r") as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines

            values = list(map(int, line.split()))
            x = values[0]
            y_values = values[1:]

            xs.extend([x] * len(y_values))
            ys.extend(y_values)

    plt.figure()
    plt.scatter(xs, ys, alpha=0.05, color='blue')

    # create points for the line
    x_fit = np.arange(min(xs), max(xs)+1)
    y_fit = [0.25 * 1e-6 * (getClassicBidiagTime(i) + getClassicQRTime(i)) for i in x_fit]
    y_fit6 = [6 * i for i in y_fit]
    
    plt.plot(x_fit, y_fit, color='purple', label='predicted')
    plt.plot(x_fit, y_fit6, color='black', label='predicted times 6')
    
    
    plt.title('Predicted vs Actual Time for GRK')
    plt.xlabel('Dimension')
    plt.ylabel('Time [ms]')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("plotGRKtimeCheck.pdf", bbox_inches="tight")
    
    plt.show()
    
    
def timecheck_QRSVD():
    xs = []
    ys = []

    with open("timeCheckQR.txt", "r") as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines

            values = list(map(int, line.split()))
            x = values[0]
            y_values = values[1:]

            xs.extend([x] * len(y_values))
            ys.extend(y_values)

    plt.figure()
    plt.scatter(xs, ys, alpha=0.05, color='blue')

    # create points for the line
    x_fit = np.arange(min(xs), max(xs)+1)
    y_fit = [0.25 * 1e-6 * getALTQRLinTime(i) for i in x_fit]
    
    plt.plot(x_fit, y_fit, color='purple', label='predicted')
    
    
    plt.title('Predicted vs Actual Time for QR')
    plt.xlabel('Dimension')
    plt.ylabel('Time [ms]')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("plotQRtimeCheck.pdf", bbox_inches="tight")
    
    plt.show()


def plot_profi_comparison(first):
    xs = []
    ys = []

    with open("svd_timings750.txt", "r") as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines

            values = list(map(float, line.split()))
            x = values[0]
            y_values = values[1:]

            xs.extend([x] * int(len(y_values) / 2))
            if first:
                ys.extend(y_values[:int(len(y_values) / 2)])
            else:
                ys.extend(y_values[int(len(y_values) / 2):])
            

    plt.figure()
    plt.scatter(xs, ys, alpha=0.05, color='blue')

    # create points for the line
    x_fit = np.arange(min(xs), max(xs)+1)
    y_fit = [0.25 * 1e-6 * (getClassicalBidiagTimeUsingGPU(i) + getClassicalQRTimeUsingGPU(i)) for i in x_fit]
    
    y_fit7 = [7 * i for i in y_fit]
    
    plt.plot(x_fit, y_fit, color='purple', label='predicted')
    plt.plot(x_fit, y_fit7, color='black', label='predicted times 7')
    
    if first:
        plt.title('Predicted vs Actual Time for LAPACK gesdd')
    else:

        plt.title('Predicted vs Actual Time for LAPACK gesvd') 
    plt.xlabel('Dimension')
    plt.ylabel('Time [ms]')
    plt.legend()
    plt.tight_layout()
    
    if first:
        plt.savefig("plotLapackSDDcomparison.pdf", bbox_inches="tight")
    else:
        plt.savefig("plotLapackSVDcomparison.pdf", bbox_inches="tight")
    
    plt.show()

def plotAll():
    plt.rcParams["font.size"] = 14
    
    timecheck_GRKSVD()
    timecheck_QRSVD()
    
    print('timecheck done')
    plot1()
    plot2()
    print('time estimates done')
    plot3()
    plot4()
    print('energy estimates done')
    plot_required_iterations_GRKSVD()
    plot_required_iterations_QRSVD()
    print('required iterations done')
    plot_profi_comparison(True)
    plot_profi_comparison(False)
    print('profi comparison done')
    
    #This takes a bit
    plotGRKSVDconvergence(15, 0, 110)
    plotQRSVDconvergence(15, 0, 110)
    
    


plotAll()
