import MPI
MPI.Init(); c=MPI.COMM_WORLD
println("rank $(MPI.Comm_rank(c))/$(MPI.Comm_size(c)) on $(gethostname())"); flush(stdout)
MPI.Barrier(c)
