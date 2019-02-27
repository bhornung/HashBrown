import numpy as np
import numba


def calculate_block_indices(series, n_blocks):
    """
    Calculates the left and rightmost indices of the blocks.
    Parameters:
        series (np.ndarray)
        n_blocks : number of blocks
    Returns:
        block_indices (np.ndarray) : the delimiting indices of the blocks.
    """
    
    if n_blocks > series.shape[1]:
        raise IndexError("Number of blocks exceeds length of array.")

    block_indices = [(idcs[0], idcs[-1]) for idcs in np.array_split(np.arange(series.shape[1]), n_blocks)]
    block_indices = np.array(block_indices, dtype = np.int)

    return block_indices

@numba.jit(nopython = True)
def calculate_pattern_hash(pattern_spec, arrlen):
    """
    Parameters:
        pattern ((int,int)) : the position and length of a pattern
        arrlen (int) : length of binary encoding array (number of blocks - 1)
    Returns:
        hash_ (int) : the hash of the pattern
    """
    
    pos, length = pattern_spec

    hash_ = (length - 1) * (2 * arrlen - (length - 2)) // 2 + pos
        
    return hash_
	
def calculate_loss_function(segmentation, store, hash_table):
    """
    Calculates the additive loss function.
    Parameters:
        segmentation (np.ndarray) : binary array representing a segmentation
        store (np.ndarray) : array to store calculated hash keys
        hash_table ({int:float}) : hash table encoding the pattern values
    Returns:
        loss (float) : the value of the loss function at segmentation
    """
    
    hash_keys = translate_segmentation(segmentation, store)
    loss = sum(hash_table[k] for k in hash_keys)
    
    return loss
	

@numba.jit(nopython = True)
def translate_segmentation(arr, store):
    """
    Translates a binary string representation of a segmentation
    to a list of hash keys.
    Parameters:
        arr (np.ndarray of 0 and 1): binary representation of a segmentation
        store (np.ndarray) : storage for hash keys
    Returns:
        store (np.ndarray) : section of store updated with the new hash keys
    """
    
    is_free = True
    length = 1
    iseg = -1
    arrlen = arr.size + 1
    
    for i, x in enumerate(arr):
        
        if x == 1:
            if is_free:
                pos = i
                is_free = False
            length += 1
        else:
            # end of subsequent ones
            if length > 1:
                hash_ = calculate_pattern_hash((pos, length), arrlen)
                length = 1
                is_free = True
            
            # add zeros
            else:
                hash_ = i
                
            iseg += 1
            store[iseg] = hash_               
    
    # last element
    if length > 1:
        hash_ = calculate_pattern_hash((pos, length), arrlen) 
    else:
        hash_ = i + 1
        
    iseg += 1
    store[iseg] = hash_
      
    return store[:iseg + 1]
	
	
class HashBrown(object):
    """
    Class to create a hash table for block merging time series.
    
    Attributes:
        n_blocks (int) : number of blocks
        series (np.ndarray) : the underlying time series
        table ({:}) : hash table of the merge costs. 
    
    Methods:
        create_pattern_generator() : returns a generator object of the patterns. 
        Each element is tuple of position and length of the pattern in ascending order.
        create_table() : creates a hash table using the merge cost and hash function 
    """
    
    @property
    def n_blocks(self):
        return self._n_blocks
   
    @property
    def series(self):
        return self._series
    
    @property
    def table(self):
        return self._table
    
    def __init__(self, series, n_blocks, 
                    func, hfunc, 
                    func_args = [], func_kwargs = {},
                    hfunc_args = [], hfunc_kwargs = {}):
        """
        n_blocks (int) : number of blocks
        func (callable) : cost of merging blocks to a pattern
        func_args ([]) : arguments of func
        func_kwargs({:}) : keyword arguments of func
 
        hfunc (callable) : hash function
        hfunc_args ([]) : arguments of hfunc
        hfunc_kwargs({:}) : keyword arguments of hfunc  
        """
        self._series = series
        
        self._n_blocks = n_blocks
        self._string_length = self.n_blocks - 1
        
        # cost function
        self._func = func
        self._func_args = func_args
        self._func_kwargs = func_kwargs
   
        # hash function
        self._hfunc = hfunc
        self._hfunc_args = hfunc_args
        self._hfunc_kwargs = hfunc_kwargs
        
        # hash table
        self._table = None

    def create_pattern_generator(self):
        """
        Creates a generator of patterns. The generator itself is a sequence
        of (position, length) tuples.
        Example: (0,1), (1,1), (2,1), ..., (0,2), (1,2), ...
        """
        
        pattern_generator = ((i_start, i_end + 1) for i_end in range(self._n_blocks) 
                                for i_start in range(self._n_blocks - i_end))
        
        return pattern_generator

    def create_table(self):
        """
        Creates a hash table of the merge costs of patterns.
        The key is the integer hash of the pattern.
        The value is the merge cost of that pattern.
        """
         
        # create iterator
        patterns = self.create_pattern_generator()
        
        # populate table
        table = {self._hfunc(pattern, *self._hfunc_args, **self._hfunc_kwargs) :
                 self._func(pattern, self._series, *self._func_args, **self._func_kwargs)
                     for pattern in patterns}
        
        self._table = table