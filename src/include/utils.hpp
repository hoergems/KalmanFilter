#ifndef _UTILS_KALMAN_
#define _UTILS_KALMAN_

namespace utils {

Eigen::VectorXd toEigenVec(std::vector<double> &vec) {	
	Eigen::Map<Eigen::VectorXd> e_vec(vec.data(), vec.size());	
	return e_vec;
	
}

std::vector<double> toStdVec(Eigen::VectorXd &vec) {
	std::vector<double> res;
	res.resize(vec.size());
	Eigen::VectorXd::Map(&res[0], vec.size()) = vec;
	return res;
}

}

#endif

