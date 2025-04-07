import numpy as np
import pandas as pd
from nfft import nfft as adjoint #CHANGE IN CONVENTION
from nfft import nfft_adjoint as nfft #CHANGE IN CONVENTION
from scipy.linalg import lu,toeplitz
#import sym_matrix

def ndft_mat(x,N):
    #non-equispaced discrete Fourier transform Matrix
    k = -(N // 2) + np.arange(N)
   
    return np.asmatrix(np.exp(2j * np.pi * np.outer(k,x[:,np.newaxis])).T)

def change_last_true_to_false(arr):
    
    arr = np.asarray(arr)
    indices = np.where(arr)[0]
    if len(indices) > 0:
        last_true_index = indices[-1]
        arr[last_true_index] = False
    
    return arr

def fjr(N):
    
    x = np.linspace(-1/2,1/2,N,endpoint=False)
    w = ((np.sin((N/2) * np.pi * x) / np.sin(np.pi * x)) ** 2) * np.divide(2*(1 + np.exp(-2 * np.pi * 1j * x)),N ** 2)
    w[x.shape[0] // 2] = 1
    
    return w

def sobg(z,a,b,g):
    
    w = np.divide((0.25 - z ** 2) ** b, g + np.abs(z) ** (2 * a))
    c = sum(abs(w)) ** (-1)
    
    return w * c

def sobk(N,a,b,g):
    
    x = np.linspace(-1/2,1/2,N,endpoint=False)
    k = np.linspace(-N/2,N/2,N,endpoint=False)
    w = sobg(k/N,a,b,g)
    s = np.divide(1 + np.exp(-2 * np.pi * 1j * x),2 * np.sum(adjoint(x,w))) * adjoint(x,w) 
    #s[x.shape[0]//2] = 1
    
    return s

def infft(x, y, N, AhA=None, w=None, return_adjoint=False, approx=False):
    
    if w is None:
        w = np.ones(N) / N
        Warning("No weight function input; normalized uniform weight for all frequencies")

    if AhA is None and approx == False:
        A = ndft_mat(x,N)
        AhA = A.H @ A
        Warning("No self-adjoint matrix specified; calculating based on input observations")
    
    if approx == False:
        L,U = lu(AhA,permute_l=True)
        fk = nfft(x,y,N) @ (np.diag(w) - np.diag(w) @ L @ np.linalg.pinv(np.eye(N) + U @ np.diag(w) @ L) @ U @ np.diag(w))
    else:
        fk = (nfft(x,y,N) @ np.diag(w)) @ np.linalg.pinv(len(x) * np.diag(w) + np.eye(N))
    
    if return_adjoint == True:
        fj = np.real(adjoint(x,fk))
        res_abs = np.sum(np.abs(y - fj) ** 2)
        res_rel = res_abs / np.sum(y ** 2)
    else:
        fj = None
        res_abs = None
        res_rel = None

    return fk, fj, res_abs, res_rel

def ndft_mat_nd(spatial_points, num_frequencies_per_dim):
    """
    Constructs the non-equispaced discrete Fourier transform (NDFT) matrix for N dimensions using matmul.

    C API is under sym_matrix.c in the core directory. This is very slow, and not recommended.

    Parameters:
        spatial_points (np.ndarray): Spatial points, shape (M, D) or (M,) for 1D.
        num_frequencies_per_dim (int or list[int]): Number of frequency points per dimension.
                                                    Can be an integer (1D) or list for N-D.

    Returns:
        np.ndarray: Transformation matrix A of shape (M, total_frequencies).
    """
    # Ensure spatial_points is a numpy array
    spatial_points = np.asarray(spatial_points)

    # Handle 1D case: If num_frequencies_per_dim is an integer, convert it to a list
    if isinstance(num_frequencies_per_dim, int):
        num_frequencies_per_dim = [num_frequencies_per_dim]

    # Handle the case where spatial_points is 1D (reshape to M x 1 for N-D compatibility)
    if spatial_points.ndim == 1:
        spatial_points = spatial_points[:, np.newaxis]  # Shape (M, 1)

    # Extract dimensions
    M, D = spatial_points.shape  # M: Number of spatial points, D: Dimensions
    if len(num_frequencies_per_dim) != D:
        raise ValueError("Length of num_frequencies_per_dim must match the dimensionality of spatial_points.")

    # Generate frequency points for each dimension
    frequency_grids = [-(N // 2) + np.arange(N) for N in num_frequencies_per_dim]
    if D == 1:  # Special case for 1D
        frequency_points = frequency_grids[0][:, np.newaxis]  # Shape (N, 1)
    else:
        frequency_points = np.array(np.meshgrid(*frequency_grids, indexing="ij")).reshape(D, -1).T  # Shape (total_frequencies, D)
    
    # Compute the transformation matrix
    # Outer product in N dimensions -> dot product between spatial and frequency points using matmul
    phase_matrix = 2j * np.pi * np.matmul(spatial_points, frequency_points.T)  # Shape (M, total_frequencies)
    transformation_matrix = np.exp(phase_matrix)
    
    return transformation_matrix

def infft_2d(data, N, AhA=None, w=None):
    """
    Perform a 2D inverse non-uniform FFT (INFFT) with operations applied along columns first.
    
    Parameters:
        data: np.ndarray
            2D input data array.
        N: int
            Length of the FFT.
        AhA: Optional
            Precomputed matrix or value for the inverse operation.
        w: np.ndarray, optional
            Weight vector for the given axis.
    
    Returns:
        tuple:
            - np.ndarray: The transformed data in 2D.
            - list: Mean values for each column.
    """
    t = np.linspace(-0.5, 0.5, data.shape[0], endpoint=False)
    h_k = -(N // 2) + np.arange(N)

    ftot_list = []
    mtot_list = []

    # Step 1: Column-wise INFFT
    for jj in range(data.shape[1]):
        idx = ~np.isnan(data[:, jj])  # Identify valid (non-NaN) values
        if np.sum(idx) == 0:
            ftot_list.append(np.zeros(N, dtype=complex))
            mtot_list.append(0)
            continue

        AAh = compute_sym_matrix_optimized(t[idx], h_k)

        if np.sum(idx) % 2 != 0:
            idx[np.where(idx)[0][-1]] = False  # Adjust to even number of samples if needed

        mn = np.mean(data[idx, jj])
        ftot, _, _, _ = infft(t[idx], data[idx, jj] - mn, N=N, AhA=AAh, w=w)

        ftot_list.append(ftot)
        mtot_list.append(mn)

    ftot = np.array(ftot_list).T
    mtot = np.array(mtot_list)

    # Step 2: Row-wise FFT
    result = np.fft.fft(ftot, axis=1)

    return result, mtot


def adjoint_transform_2d(transformed_data, mtot, data_shape):
    """
    Perform the adjoint of the 2D forward transform to reconstruct the original data, with operations along rows.
    
    Parameters:
        transformed_data: np.ndarray
            2D transformed data array.
        mtot: list
            Mean values for each column from the forward transform.
        data_shape: tuple
            Shape of the original data (to restore NaNs).
    
    Returns:
        np.ndarray
            Reconstructed data including NaNs in their original locations.
    """
    t = np.linspace(-0.5, 0.5, data_shape[0], endpoint=False)

    reconstructed_data = np.full(data_shape, np.nan, dtype=np.float64)  # Initialize with NaNs

    # Step 1: Inverse row-wise FFT
    iftot = np.fft.ifft(transformed_data, axis=1)

    # Step 2: Column-wise adjoint transform
    for jj in range(data_shape[1]):
        # Perform the adjoint operation
        adjoint_result = adjoint(t, iftot[:, jj])

        # Restore mean and NaNs
        reconstructed_column = adjoint_result + mtot[jj]
        reconstructed_data[:, jj] = np.abs(reconstructed_column)

    return reconstructed_data

def compute_sym_matrix_optimized(f_j, h_k):
    """
    Compute the inner product matrix when h_k is equally spaced.
    That is, compute the Toeplitz matrix A with
        A[k1, k2] = sum_j exp(2πi * f_j * (h_k[k2] - h_k[k1])),
    where h_k[k] = h0 + k*d.
    """
    f_j = np.asarray(f_j, dtype=float)
    h_k = np.asarray(h_k, dtype=float)
    N = h_k.size

    # Verify that h_k is equally spaced.
    d = h_k[1] - h_k[0]
    if not np.allclose(np.diff(h_k), d):
        raise ValueError("h_k must be equally spaced for a Toeplitz structure.")

    # Compute the unique values for the first column.
    # The lag for entry (0, k) is h_k[k] - h_k[0] = d * k.
    lags = d * np.arange(N)
    col = np.array([np.sum(np.exp(-2 * np.pi * 1j * f_j * lag)) for lag in lags])

    # For a Toeplitz matrix, the entry A[i,j] depends only on (j-i).
    # Since A[0,j] is given by 'col', we build the full matrix.
    A = toeplitz(col)
    
    # For numerical precision, enforce that the diagonal is exactly M.
    # Here, on the diagonal, lag=0 so exp(0)=1, and sum_j 1 = len(f_j).
    np.fill_diagonal(A, len(f_j))
    
    return A

