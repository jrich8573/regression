# install.packages("matrixcalc")
require(Matrix)
require(matrixcalc)

# A.  Steepest Descent

SteepDescent <- function(A = NULL, b = NULL, x0 = NULL, iter_num = 100, sig_thres = 1e-5){

  ##  required packages
  require(Matrix)       # func is.symmetric.matrix()
  require(matrixcalc)   # func is.positive.definite()

  ## some pre-condition
  if(is.null(A) | is.null(b) | is.null(x0)){
    stop("Please input the correct version of A, b, or x0! \n")
  }

  if((!is.positive.definite(A) | !is.symmetric.matrix(A)) ){
    stop("Sorry, the A matrix is not a symmetric, positive-definite matrix! \n")
  }

  if(ncol(A) != length(x0) | nrow(A) != length(b)){
    stop("Sorry, the dim of A, b, or x0 does not match. Please check you input matrix and vectors! \n")
  }

 ###############################################################################
 ## the actual algorithm:
 ###############################################################################

  r <- b - A %*% x0
  residual_origin <- sum(r*r)

  r_old <- r
  x_old <- x0


  # iteration
  for(i in 1:iter_num){
    q <- A %*% r_old
    residual <- sum(r_old*r_old)
    alpha <- residual/as.numeric(t(r_old) %*% q)

    x_new <- x_old + alpha * r_old

    if(i %% 20 == 0){
      r_new <- b - A %*% x_old
    }else{
      r_new <- r_old - alpha * q
    }

    x_old <- x_new
    r_old <- r_new
    rm(x_new, r_new)

    if(residual <= sig_thres^2 * residual_origin){
      break
    }
  }

  if(i == iter_num & residual > sig_thres^2 * residual_origin){
    message(paste("Within the", iter_num, "iteration time, the results did not converge to the sig threshold", sig_thres, "!"))
  }
  cat(paste("the iteration number is", i, "!\n"))
  return(x_old)

}



ConjugateGradient <- function(A = NULL, b = NULL, x0 = NULL, iter_num = 100, sig_thres = 1e-5){
  ##  required packages
  require(Matrix)       # func is.symmetric.matrix()
  require(matrixcalc)   # func is.positive.definite()

  ## some pre-condition
  if(is.null(A) | is.null(b) | is.null(x0)){
    stop("Please input the correct version of A, b, or x0! \n")
  }

  if((!is.positive.definite(A) | !is.symmetric.matrix(A)) ){
    stop("Sorry, the A matrix is not a symmetric, positive-definite matrix! \n")
  }

  if(ncol(A) != length(x0) | nrow(A) != length(b)){
    stop("Sorry, the dim of A, b, or x0 does not match. Please check you input matrix and vectors! \n")
  }

################################################################################
 ## the actual algorithm:
################################################################################

  r <- b - A %*% x0
  d <- r

  residual_origin <- sum(r*r)

  r_old <- r
  x_old <- x0
  residual_old <- residual_origin

  ## iteration
  for(i in 1:iter_num){
    q <- A %*% d
    alpha <- residual_old/as.numeric(t(d) %*% q)

    x_new <- x_old + alpha*d
    if(i %% 20 == 0){
      r_new <- b - A %*% x_old
    }else{
      r_new <- r_old - alpha * q
    }

    residual_new <- sum(r_new*r_new)
    beta <- residual_new/residual_old
    d <- r_new + beta*d

    x_old <- x_new
    r_old <- r_new
    residual_old <- residual_new

    rm(x_new, r_new, residual_new)

    if(residual_old <= sig_thres^2 * residual_origin){
      break
    }

  }


  if(i == iter_num & residual_old > sig_thres^2 * residual_origin){
    message(paste("Within the", iter_num, "iteration time, the results did not converge to the sig threshold", sig_thres, "!"))
  }
  cat(paste("the iteration number is", i, "!\n"))
  return(x_old)


}





##   ### This is a test data
##   T_m <- matrix(c(1, 2, 1, 3, 5, 9, 2, 4, 1, 0, 5, 1), nrow=4)
##   A <- t(T_m) %*% T_m   # make sure A is symmetric and positive-definite
##   b <- c(2, 2 ,2)
##   x0 <- c(5, 5, 5)
##
##   solve(A) %*% b
##
##   SteepDescent(A, b, x0)
##   system.time(SteepDescent(A, b, x0))
##   ## 42 iterations needed.
##
##   ConjugateGradient(A, b, x0)
##   system.time(ConjugateGradient(A, b, x0))
##   ## Only 3 iterations needed!! Super fast.



