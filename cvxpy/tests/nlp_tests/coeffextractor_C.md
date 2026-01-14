❯ I have an idea and need your help planning through it. While working on the differentiation           
  engine, I noticed that there are lots of similarities between implementations of affine atoms and     
  those in CVXPY's canonicalization                                                                     
  backend'/Users/willizz/Documents/DNLP/cvxpy/lin_ops/canon_backend.py'. After some thought, I          
  finally think I understand the similarities. Given a list of expressions for the constraints, the     
  canon backend essentially computes its constant Jacobian. One additional thing that is supported      
  in cvxpy though is the ability to include parameters in the expression trees; we will ignore          
  this additional feature and only focus on problems without parameters.\                               
  '/Users/willizz/Documents/DNLP/cvxpy/utilities/coeff_extractor.py' this file shows where the          
  canon backend is being called, it is the affine function. \                                           
  I want you to do the following: replace this canonInterface.get_problem_matrix function with the      
  differentiation_engine (calling its jacobian function) and try to ensure that cvxpy canonicalizes     
  correctly for some toy problems. \                                                                    
  Then eventually I would like you to run benchmarks in the benchmark repository. Please skip all       
  benchmarks which have parameters in them. Then it would be nice to compare and see if this            
  differentiation engine in C is faster than the canonbackend implementations written in scipy. \       
  Can you comment and give your perspective on all of the above? First tell me if this connection       
  is correct. Then secondly, tell me if there are more details you need to start the                    
  implementations. For example, get_problem_matrix might assume that one dimension is for               
  parameters, so you might need to reshape the output to get the desired A matrix.                      
  '/Users/willizz/Documents/DNLP/cvxpy/tests/nlp_tests/coeffextractor_test.py' this file shows how      
  to use the CoeffExtractor. You do need to reshape the output to a (num_constraints,                   
  num_variables) matrix in Fortran order.