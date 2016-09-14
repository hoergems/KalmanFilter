#ifndef _UTILS_KALMAN_
#define _UTILS_KALMAN_

namespace utils_kalman {

Eigen::VectorXd toEigenVec(std::vector<double> &vec) {	
	Eigen::Map<Eigen::VectorXd> e_vec(vec.data(), vec.size());	
	return e_vec;
	
}

std::vector<double> addVectors(std::vector<double> &vec1, std::vector<double> &vec2) {
	std::vector<double> res(vec1.size());
	for (size_t i = 0; i < vec1.size(); i++) {
		res[i] = vec1[i] + vec2[i];
	}
	
	return res;
}

std::vector<double> subtractVectors(std::vector<double> &vec1, std::vector<double> &vec2) {
	std::vector<double> res(vec1.size());
	for (size_t i = 0; i < vec1.size(); i++) {
		res[i] = vec1[i] - vec2[i];
	}
	
	return res;
}

std::vector<double> toStdVec(Eigen::VectorXd &vec) {
	std::vector<double> res;
	res.resize(vec.size());
	Eigen::VectorXd::Map(&res[0], vec.size()) = vec;
	return res;
}

}

#endif

