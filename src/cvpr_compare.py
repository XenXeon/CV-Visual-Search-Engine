import numpy as np
import math

def cvpr_compare(F1, F2, metric='L2', eigenvalues=None):
    # This function should compare F1 to F2 - i.e. compute the distance
    # between the two descriptors

    if metric == 'L2':
        x=F1-F2
        x=x**2
        x=sum(x)
        dst = math.sqrt(x)
        return dst
    
    elif metric == 'L1':
        x=F1-F2
        dst = np.sum(np.abs(x))
        return dst
    
    elif metric == 'L3':
        x=F1-F2
        dst = np.power(np.sum(np.abs(x)**3), 1/3)
        return dst
    
    elif metric == 'Mahalanobis':
        
        if eigenvalues is None:
            raise ValueError("Eigenvalues must be provided for Mahalanobis distance.")
        
        # Ensure F1 and eigenvalues have the same shape
        if F1.shape[0] != eigenvalues.shape[0]:
            print(f"Warning: Feature vector length ({F1.shape[0]}) does not match "
                  f"eigenvalues length ({eigenvalues.shape[0]}).")
            # Truncate to the smallest
            min_len = min(F1.shape[0], eigenvalues.shape[0])
            F1 = F1[:min_len]
            F2 = F2[:min_len]
            eigenvalues = eigenvalues[:min_len]

        # Add a tiny value (epsilon) to eigenvalues to avoid division by zero
        # [cite_start]The lecture slides show eigenvalues can be 0.0000 [cite: 151-159]
        epsilon = 1e-10
        safe_eigenvalues = eigenvalues + epsilon
        
        diff = F1 - F2
        
        # [cite_start]Formula from slide 14: d = sqrt( sum( (q_i - t_i)^2 / v_i^2 ) ) [cite: 211]
        scaled_diff = (diff**2) / safe_eigenvalues
        
        dst = np.sqrt(np.sum(scaled_diff))
        return dst
        
    else:
        raise ValueError(f"Unknown metric type: {metric}.")
