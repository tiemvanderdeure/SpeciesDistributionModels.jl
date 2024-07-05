struct NoResampling <: MLJBase.ResamplingStrategy end
MLJBase.train_test_pairs(::NoResampling, indices, _)  = [(indices, eltype(indices)[])]## get indices

struct CustomRows <: MLJBase.ResamplingStrategy end
