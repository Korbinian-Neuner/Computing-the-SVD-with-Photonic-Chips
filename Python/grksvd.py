import numpy as np
import sys
import queue
import svdviaqr


# Computes Householder vector x for column k (left transformation)
# Used during bidiagonalization to zero out entries below the diagonal
def calcHouseholderX(matrixA, k):
    m = matrixA.shape[0]
    n = matrixA.shape[1]
    
    # Compute norm of column k from row k downward
    s = float(0)
    for i in range(k, m):
        s = s + matrixA[i, k] * matrixA[i, k]
    s = np.sqrt(s)
    
    if np.isclose(s, 0.0, atol=1e-14):
        # vector is already zero, return identity
        return np.zeros((m, 1))
    
    # Compute reflection scalar alpha
    alpha = -s * np.sign(matrixA[k, k])
    
    # First component of Householder vector
    x_k = float(1) + np.abs(matrixA[k, k]) / s
    x_k = x_k / 2
    x_k = np.sqrt(x_k)
    
    # Scaling constant
    c = float(1) / (-2 * alpha * x_k)  #rearranged from reference in GRK paper
    
    # Build full Householder vector
    x = np.zeros((m, 1))
    x[k, 0] = x_k
    for i in range(k + 1, m):
        x[i, 0] = c * matrixA[i, k]
    
    return x

# Computes Householder vector y for row k (right transformation)
# Used during bidiagonalization to zero out entries right of superdiagonal
def calcHouseholderY(matrixA, k):
    m = matrixA.shape[0]
    n = matrixA.shape[1]
    
    # Compute norm of row k from column k+1 onward
    t = float(0)
    for j in range(k+1, n):
        t = t + matrixA[k, j] * matrixA[k, j]
    t = np.sqrt(t)
    
    if np.isclose(t, 0.0, atol=1e-14):
        # vector is already zero, return identity
        return np.zeros((n, 1))
    
    # Compute reflection scalar beta
    beta = -t * np.sign(matrixA[k, k + 1])
    
    # First component of Householder vector
    y_k = float(1) + np.abs(matrixA[k, k + 1]) / t #is y_k+1 in paper
    y_k = y_k / 2
    y_k = np.sqrt(y_k)
    
    # Scaling constant
    d = float(1) / (-2 * beta * y_k)  #also rearranged
    
    # Build full Householder vector
    y = np.zeros((n, 1))
    y[k + 1, 0] = y_k
    for j in range(k + 2, n):
        y[j, 0] = d * matrixA[k, j]
    
    return y

# Reduces matrixA to bidiagonal form using Householder reflections
# Returns orthogonal matrices P, Q and bidiagonal matrix
def bidiagonalize(matrixA):
    m = matrixA.shape[0]
    n = matrixA.shape[1]
    
    P = np.eye(m, m) # Accumulates left transformations
    Q = np.eye(n, n) # Accumulates right transformations
    
    # Apply alternating left and right Householder transformations
    for k in range(n - 1):
        x = calcHouseholderX(matrixA, k)
        matrixA = matrixA - 2 * np.dot(x, np.dot(x.T, matrixA))
        P = np.dot(P, np.eye(m, m) - 2 * np.dot(x, x.T))
        y = calcHouseholderY(matrixA, k)
        matrixA = matrixA - 2 * np.dot(np.dot(matrixA, y), y.T)
        Q = np.dot(np.eye(n, n) - 2 * np.dot(y, y.T), Q)
    
    # Final left transformation
    x = calcHouseholderX(matrixA, n - 1)
    matrixA = matrixA - 2 * np.dot(x, np.dot(x.T, matrixA))
    P = np.dot(P, np.eye(m, m) - 2 * np.dot(x, x.T))
    
    
    return [P, matrixA, Q]

# Computes Wilkinson shift for implicit QR step on bidiagonal matrix
def calc_shift(matrixA):
    m = matrixA.shape[0]
    n = matrixA.shape[1]
    
    # Elements of trailing 2x2 block of B^T B
    a = matrixA[n - 3, n - 2] * matrixA[n - 3, n - 2] + matrixA[n - 2, n - 2] * matrixA[n - 2, n - 2]
    b = matrixA[n - 2, n - 1] * matrixA[n - 2, n - 2]
    c = matrixA[n - 2, n - 1] * matrixA[n - 2, n - 1] + matrixA[n - 1, n - 1] * matrixA[n - 1, n - 1]
    
    # Wilkinson eigenvalue shift. If statement to return largest eigenvalue
    r = np.sqrt((a - c) * (a - c) + b * b)
    if a > c:
        return (a + c - r) / 2
    else:
        return (a + c + r) / 2
        

# Reconstructs orthogonal matrix from stored Givens rotation parameters
def reconstruct_givens(data, dim):
    givens = np.eye(dim)
    for k in range(data.shape[0]):
        x = data[k, 0]
        c = 1 / np.sqrt(1 + x*x)
        s = c * x
        
        # Apply rotation to accumulated matrix
        for i in range(k + 2):
            n00 =  c * givens[i, k] + s * givens[i, k + 1]
            n01 = -s * givens[i, k] + c * givens[i, k + 1]

            givens[i, k] = n00
            givens[i, k + 1] = n01
    return givens


# Performs one implicit QR step on bidiagonal matrix
# Optionally returns accumulated left (S) and right (T) rotations
def qr_step(matrixA, withS = False, withT = False):
    m = matrixA.shape[0]
    n = matrixA.shape[1]
    
    # Storage for Givens rotation parameters
    stoT = np.ones((n - 1, 1))
    stoS = np.ones((n - 1, 1))
    
    shift = calc_shift(matrixA)
    
    # First Givens rotation using shift
    x = matrixA[0, 1] * matrixA[0, 0] / (matrixA[0, 0] * matrixA[0, 0] - shift)
    stoT[0, 0] = x
    c = 1 / np.sqrt(1 + x*x)
    s = c * x
    
    # Apply rotation to leading 2x2 block
    n00 =  c * matrixA[0, 0] + s * matrixA[0, 1]
    n10 =  s * matrixA[1, 1]
    n01 = -s * matrixA[0, 0] + c * matrixA[0, 1]
    n11 =  c * matrixA[1, 1]
    
    matrixA[0, 0] = n00
    matrixA[1, 0] = n10
    matrixA[0, 1] = n01
    matrixA[1, 1] = n11
    
    # Bulge chasing
    for k in range(1, n):
        
        # Right rotations (T)
        if k > 1:
            #Handle Near-Zero Edgecase
            c = 1.0
            s = 0.0
            stoT[k - 1, 0] = 0.0
            if not np.isclose(matrixA[k - 2, k - 1], 0.0, atol=1e-14):
                x = matrixA[k - 2, k] / matrixA[k - 2, k - 1]
                stoT[k - 1, 0] = x
                c = 1 / np.sqrt(1 + x*x)
                s = c * x
            
            n00 =  c * matrixA[k - 2, k - 1] + s * matrixA[k - 2, k]
            n10 =  c * matrixA[k - 1, k - 1] + s * matrixA[k - 1, k]
            n20 =  c * matrixA[k    , k - 1] + s * matrixA[k    , k]
            n01 = -s * matrixA[k - 2, k - 1] + c * matrixA[k - 2, k]
            n11 = -s * matrixA[k - 1, k - 1] + c * matrixA[k - 1, k]
            n21 = -s * matrixA[k    , k - 1] + c * matrixA[k    , k]
            
            matrixA[k - 2, k - 1] = n00
            matrixA[k - 1, k - 1] = n10
            matrixA[k    , k - 1] = n20
            matrixA[k - 2,     k] = n01
            matrixA[k - 1,     k] = n11
            matrixA[k    ,     k] = n21
            
        # Left rotations (S)
        if k < (n - 1):
            #Handle Near-Zero Edgecase
            c = 1.0
            s = 0.0
            stoS[k - 1, 0] = 0.0
            if not np.isclose(matrixA[k - 1, k - 1], 0.0, atol=1e-14):
                x = matrixA[k, k - 1] / matrixA[k - 1, k - 1]
                stoS[k - 1, 0] = x
                c = 1 / np.sqrt(1 + x*x)
                s = c * x
            
            n00 =  c * matrixA[k - 1, k - 1] + s * matrixA[k    , k - 1]
            n01 =  c * matrixA[k - 1, k    ] + s * matrixA[k    , k    ]
            n02 =  c * matrixA[k - 1, k + 1] + s * matrixA[k    , k + 1]
            n10 = -s * matrixA[k - 1, k - 1] + c * matrixA[k    , k - 1]
            n11 = -s * matrixA[k - 1, k    ] + c * matrixA[k    , k    ]
            n12 = -s * matrixA[k - 1, k + 1] + c * matrixA[k    , k + 1]
            
            matrixA[k - 1, k - 1] = n00
            matrixA[k - 1, k    ] = n01
            matrixA[k - 1, k + 1] = n02
            matrixA[k    , k - 1] = n10
            matrixA[k    , k    ] = n11
            matrixA[k    , k + 1] = n12
    
    # Final rotation to eliminate last subdiagonal
    c = 1.0
    s = 0.0
    stoS[n - 2, 0] = 0.0
    if not np.isclose(matrixA[n - 2, n - 2], 0.0, atol=1e-14):
        x = matrixA[n - 1, n - 2] / matrixA[n - 2, n - 2]
        stoS[n - 2, 0] = x
        c = 1 / np.sqrt(1 + x*x)
        s = c * x
    
    n00 =  c * matrixA[n - 2, n - 2] + s * matrixA[n - 1, n - 2]
    n10 = -s * matrixA[n - 2, n - 2] + c * matrixA[n - 1, n - 2]
    n01 =  c * matrixA[n - 2, n - 1] + s * matrixA[n - 1, n - 1]
    n11 = -s * matrixA[n - 2, n - 1] + c * matrixA[n - 1, n - 1]
    
    matrixA[n - 2, n - 2] = n00
    matrixA[n - 1, n - 2] = n10
    matrixA[n - 2, n - 1] = n01
    matrixA[n - 1, n - 1] = n11
    
    # Return requested accumulated orthogonal matrices
    if withS and withT:
        S = reconstruct_givens(stoS, m)
        T = reconstruct_givens(stoT, n)
        return [S, matrixA, T.T]
    if withS:
        S = reconstruct_givens(stoS, m)
        return [S, matrixA, 0]
    if withT:
        T = reconstruct_givens(stoT, n)
        return [0, matrixA, T.T]
    return [0, matrixA, 0]

# Checks for deflation (splitting matrix into smaller blocks)
# If diagonal entries are small, performs cleanup rotations
def splitcheck(A, delta, withWriteUp):
    m = A.shape[0]
    n = A.shape[1]
    writeUp = []
    
    # Deflation step: remove small diagonal entries
    for k in range(n):
        if abs(A[k, k]) < delta:
            stoS = []
            
            # Apply Givens rotations to clean small diagonal entry
            for i in range(k + 1, n - 1):
                # Compute rotation to eliminate A[k, i]
                x = A[k, i] / A[i, i]
                c = 1 / np.sqrt(1 + x*x)
                s = c * x
                stoS.append(x)
                
                # Apply rotation to affected entries
                n00 = c * A[k, k]
                n10 = s * A[k, k]
                n01 = c * A[k, i] - s * A[i, i]
                n11 = s * A[k, i] + c * A[i, i]
                n02 = -s * A[i, i + 1]
                n12 = c * A[i, i + 1]
                
                A[k,k] = n00
                A[i,k] = n10
                A[k,i] = n01
                A[i,i] = n11
                A[k,i + 1] = n02
                A[i,i + 1] = n12
            
            # Final rotation for last column
            x = A[k, n - 1] / A[n - 1, n - 1]
            c = 1 / np.sqrt(1 + x*x)
            s = c * x
            stoS.append(x)
            
            n00 = c * A[k, k]
            n10 = s * A[k, k]
            n01 = c * A[k, n - 1] - s * A[n - 1, n - 1]
            n11 = s * A[k, n - 1] + c * A[n - 1, n - 1]
            
            A[k,k] = n00
            A[n - 1,k] = n10
            A[k,n - 1] = n01
            A[n - 1,n - 1] = n11
            
            # Store rotation parameters for later reconstruction
            writeUp.append(stoS)
    
    # Optionally reconstruct accumulated orthogonal matrix
    S = 0
    if withWriteUp:
        S = np.eye(m)
        for stoS in writeUp:
            k = n - 1 - len(stoS)
            S_t = np.eye(m)
            for i in range(len(stoS)):
                x = stoS[i]
                c = 1 / np.sqrt(1 + x*x)
                s = c * x
                
                # Apply stored rotations
                for j in range(k, n - 1):
                    nj0 =  c * S_t[j, k] + s * S_t[j, k + i + 1]
                    nj1 = -s * S_t[j, k] + c * S_t[j, k + i + 1]
                    S_t[j, k] = nj0
                    S_t[j, k + i + 1] = nj1
            #S = np.dot(S, S_t)
        S = S.T
    
    # Split matrix into independent diagonal blocks
    ret = []
    lastBreakIndex = 0
    for k in range(n - 1):
        # If superdiagonal element is small → split
        if abs(A[k, k + 1]) < delta:
            M = A[lastBreakIndex : k + 1, lastBreakIndex : k + 1]
            ret.append((M, lastBreakIndex))
            lastBreakIndex = k + 1
    
    # Append final block
    M = A[lastBreakIndex : n, lastBreakIndex : n]
    ret.append((M, lastBreakIndex))
        
    return (ret, S)

# Multiplies block-diagonal matrices efficiently
# Used during reconstruction of U and V
def diag_block_mul(smallBlocks, matrixA, inverted):
    offsets = []
    offset = 0
    
    # Compute block offsets
    for sb in smallBlocks:
        offsets.append(offset)
        offset = sb.shape[0] + offset
    offsets.append(offset)
    
    # Ensure dimension consistency
    assert matrixA.shape[0] == offset, str(matrixA.shape[0]) + '    ' + str(offset)
    
    choppedResult = []
    
    # Perform blockwise multiplication
    for k in range(len(offsets) - 1):
        row = []
        
        for i in range(len(offsets) - 1):
            sm = matrixA[offsets[k] : offsets[k + 1], offsets[i] : offsets[i + 1]]
            if inverted:
                row.append(np.dot(sm, smallBlocks[i]))
            else:
                row.append(np.dot(smallBlocks[k], sm))
        choppedResult.append(row)
    
    # Reassemble full matrix from blocks
    result = np.zeros(matrixA.shape)
    for x in range(len(choppedResult)):
        for y in range(len(choppedResult)):
            sm = choppedResult[x][y]
            xo = offsets[x]
            yo = offsets[y]
            for i in range(len(sm)):
                for j in range(len(sm[i])):
                    if not result[xo + i][yo + j] == 0:
                        print('ALAAAARM')
                    result[xo + i][yo + j] = sm[i][j]
    
    return result

# Ensures missing diagonal blocks are filled with identity blocks
def fixup_diag_blocks(blocks, maxSize):
    ret = []
    
    expO = 0
    for i in range(len(blocks)):
        x = blocks[i]
        # Insert identity blocks if gaps exist
        while expO < x[1]:
            ret.append((np.eye(1), expO))
            expO += 1
        ret.append(x)
        expO += x[0].shape[0]
        
    # Fill remaining tail with identities
    while expO < maxSize:
        ret.append((np.eye(1), expO))
        expO += 1
    return ret


# Full SVD computation using:
# 1. Bidiagonalization
# 2. Implicit QR iterations with deflation
# 3. Reconstruction of U and V
def svd(matrixA, withU = False, withV = False, maxIterations = 0):
    m = matrixA.shape[0]
    n = matrixA.shape[1]
    
    transposed = False
    
    # Ensure m >= n (algorithm assumes tall matrix)
    if m < n:
        matrixA = matrixA.T
        m = matrixA.shape[0]
        n = matrixA.shape[1]
        transposed = True
        
        # Swap U/V flags accordingly
        temp = withU
        withU = withV
        withV = temp
    
    # Step 1: Reduce to bidiagonal form
    P, BDR, Q = bidiagonalize(matrixA)
    BD = BDR[0:BDR.shape[1], 0:BDR.shape[1]]
    
    # Convergence threshold
    delta = max(matrixA.flat[matrixA.argmax()], 1) * sys.float_info.epsilon # according to GRK paper
    #delta = 10e-6 # Used in the paper as photonic chips have currently not reached a higher level of precision
    
    # Queue for recursive block processing
    matrixQueue = queue.SimpleQueue()
    matrixQueue.put((BD, 0, 0))
    
    stoS = [] # Stores left transformations
    stoT = [] # Stores right transformations
    
    svds = np.zeros((BD.shape[1], 1))
    
    # Default iteration cap
    if maxIterations == 0:
        maxIterations = 30 * BD.shape[0] * BD.shape[0]
    
    errors = []
    
    currentIteration = 1
    
    # Main QR iteration loop with deflation
    while not matrixQueue.empty():
        A, offset, level = matrixQueue.get()
        
        # Track diagonality error for diagnostics
        errors.append((svdviaqr.diagonality(A), level))
        
        qr_res = qr_step(A, withU, withV)
        
        A = qr_res[1]
        sc_res = splitcheck(A, delta, withU)
        
        # Store transformations for later reconstruction
        if(withU):
            stoS.append((qr_res[0], offset, level))
            stoS.append((sc_res[1], offset, level + 1))
        if(withV):
            stoT.append((qr_res[2], offset, level))
        
        # Process resulting blocks
        for x in sc_res[0]:
            M = x[0]
            
            # If 1x1 block → singular value found
            if M.shape[0] == 1:
                svds[offset + x[1]] = M[0, 0]
            else:
                matrixQueue.put((M, offset + x[1], level + 2))
                
        # Stop if iteration limit reached
        if currentIteration >= maxIterations and maxIterations > 0:
            print('Reached maximum number of iterations')
            return -1
        currentIteration += 1
    
    iterations = currentIteration
    
    # Make singular values positive
    n = len(svds)
    signs = np.sign(svds)
    svds = svds * signs
    svds = np.array(svds).flatten()
    
    # Sort singular values descending
    sort_indices = np.argsort(-svds)
    M_s = np.eye(len(svds))[sort_indices] #Permutationmatrix
    sorted_svds = np.dot(svds, M_s.T)
    
    # Compute per-iteration error metrics
    ErrorIterations = int(max(x[1] for x in errors) / 2)
    actualErrors = []
    for i in range(ErrorIterations):
        maxCurrentError = max(x[0] for x in errors if x[1] == 2 * i)
        actualErrors.append(maxCurrentError)
    
    U = 0
    V = 0
    
    # Reconstruction of U (left singular vectors)
    if withU:
        lvls = []
        
        # Group stored S-transformations by recursion level
        for x in stoS:
            while len(lvls) < (x[2] + 1):
                lvls.append([])
            lvls[x[2]].append((x[0], x[1]))
        
        # Sort blocks by offset and insert identity gaps
        for k in range(len(lvls)):
            lvls[k] = sorted(lvls[k], key = lambda x: x[1])
            lvls[k] = fixup_diag_blocks(lvls[k], n)
            
        # Merge transformations bottom-up (from deepest level upward)
        for k in range(len(lvls) - 1, 0, -1):
            smallBlocks = lvls[k]
            bigBlocks = lvls[k - 1]
            
            
            for i in range(len(bigBlocks)):
                sbs = []
                bb = bigBlocks[i]
                
                # Collect sub-blocks belonging to current big block
                for sb in smallBlocks:
                    if sb[1] >= bb[1] and sb[1] < (bb[1] + bb[0].shape[0]):
                        sbs.append(sb[0])
                
                # Multiply block-diagonal transformations
                lvls[k - 1][i] = (diag_block_mul(sbs, bb[0], True), bb[1])
        
        # Final accumulated left orthogonal factor
        G = lvls[0][0][0]
        
        # Apply sign corrections to match positive singular values
        G = np.dot(G, np.diag(signs.flatten()))
        
        # Apply permutation (sorting of singular values)
        G = np.dot(G, M_s.T)
        
        # Embed into full-size identity (since BD may be smaller than P)
        I = np.eye(P.shape[0])
        I[0:G.shape[0], 0:G.shape[1]] = G
        
        # Final U = P * accumulated rotations
        U = np.dot(P, I)
        
        # If V not requested, return early
        if not withV:
            if transposed:
                return (sorted_svds, U.T)
            return (U, sorted_svds)
    
    # Reconstruction of V (right singular vectors)
    if withV:
        lvls = []
        
        # Group stored T-transformations by recursion level
        for x in stoT:
            while len(lvls) < (x[2] / 2 + 1):
                lvls.append([])
            lvls[int(x[2] / 2)].append((x[0], x[1]))
        
        # Sort and fill identity blocks where needed
        for k in range(len(lvls)):
            lvls[k] = sorted(lvls[k], key = lambda x: x[1])
            lvls[k] = fixup_diag_blocks(lvls[k], n)
        
        # Merge transformations bottom-up
        for k in range(len(lvls) - 1, 0, -1):
            smallBlocks = lvls[k]
            bigBlocks = lvls[k - 1]
            
            for i in range(len(bigBlocks)):
                sbs = []
                bb = bigBlocks[i]
                
                # Collect sub-blocks belonging to current big block
                for sb in smallBlocks:
                    if sb[1] >= bb[1] and sb[1] < (bb[1] + bb[0].shape[0]):
                        sbs.append(sb[0])
                
                # Multiply block-diagonal transformations
                lvls[k - 1][i] = (diag_block_mul(sbs, bb[0], False), bb[1])
        
        # Final accumulated right orthogonal factor
        H = lvls[0][0][0]
        
        # Combine with initial Q from bidiagonalization
        V = np.dot(H, Q)
        
        # Apply permutation from singular value sorting
        V = np.dot(M_s, V)
        
        # If U not requested, return early
        if not withU:
            if transposed:
                return (V.T, sorted_svds)
            return (sorted_svds, V)
    
    # If both U and V requested
    if withU and withV:
        if transposed:
            return (V.T, sorted_svds, U.T)
        return (U, sorted_svds, V)
    
    # Default return: singular values, iteration count, and error history
    return sorted_svds, iterations, actualErrors


#Testing Code
def test_svd():
    np.random.seed(42)

    test_matrices = [
        np.random.randn(5, 5),
        np.random.randn(8, 5),
        np.random.randn(5, 8),
        np.diag([5, 4, 3, 2, 1]),
        np.random.randn(10, 10) * 1e-3  # small scale test
    ]

    for idx, A in enumerate(test_matrices):
        print(f"\nTest case {idx + 1}: shape = {A.shape}")

        # Our SVD
        U, s, V = svd(A, withU=True, withV=True)

        # NumPy SVD for reference
        U_np, s_np, Vt_np = np.linalg.svd(A, full_matrices=True)

        # Build Sigma
        Sigma = np.zeros_like(A, dtype=float)
        min_dim = min(A.shape)
        Sigma[:min_dim, :min_dim] = np.diag(s)

        # Reconstruction
        A_reconstructed = U @ Sigma @ V

        reconstruction_error = np.linalg.norm(A - A_reconstructed)
        singular_value_error = np.linalg.norm(np.sort(s)[::-1] - np.sort(s_np)[::-1])

        # Orthogonality checks
        U_orth_error = np.linalg.norm(U.T @ U - np.eye(U.shape[1]))
        V_orth_error = np.linalg.norm(V @ V.T - np.eye(V.shape[0]))

        print(f"Reconstruction error:      {reconstruction_error:.3e}")
        print(f"Singular value error:      {singular_value_error:.3e}")
        print(f"U orthogonality error:     {U_orth_error:.3e}")
        print(f"V orthogonality error:     {V_orth_error:.3e}")

        if reconstruction_error < 1e-6 and singular_value_error < 1e-6:
            print("Status: PASS")
        else:
            print("Status: CHECK")


if __name__ == "__main__":
    test_svd()