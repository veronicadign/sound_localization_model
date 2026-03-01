import nest

def connect(
    pre,
    post,
    conn_spec=None,
    syn_spec=None,
    return_synapsecollection=False,
    num_sources=None,
):
    if conn_spec == "x_to_one":
        if num_sources is None:
            raise Exception(
                "when using custom conn_spec 'x_to_one' argument 'num_sources' must be set"
            )
        
        num_pre = len(pre)
        num_post = len(post)
        
        # Calculate the step size - can be fractional
        step = (num_pre - num_sources) / (num_post - 1) if num_post > 1 else 0
        
        for i in range(num_post):
            # Calculate start index (can use fractional step)
            start_idx = int(round(i * step))
            end_idx = start_idx + num_sources
            
            # Ensure we don't exceed bounds
            if end_idx > num_pre:
                end_idx = num_pre
                start_idx = num_pre - num_sources
            
            nest.Connect(
                pre[start_idx:end_idx],
                post[i],
                "all_to_all",
                syn_spec,
                return_synapsecollection,
            )
    else:
        nest.Connect(pre, post, conn_spec, syn_spec, return_synapsecollection)