#include "include/kalman_filter.hpp"

using std::cout;
using std::endl;

namespace shared {

KalmanFilter::KalmanFilter() {
	
}

void KalmanFilter::kalmanPredict(Eigen::VectorXd &x, 
                                 Eigen::VectorXd &u,
                                 Eigen::MatrixXd &A,
                                 Eigen::MatrixXd &B,
                                 Eigen::MatrixXd &P_t,
                                 Eigen::MatrixXd &V,
                                 Eigen::MatrixXd &M,
                                 Eigen::VectorXd &x_predicted,
                                 Eigen::MatrixXd &P_predicted) {	
	x_predicted = A * x + B * u;	
	computePredictedCovariance(A, P_t, V, M, P_predicted);	
}

void KalmanFilter::kalmanPredictPy(std::vector<double> &x,
			                       std::vector<double> &u,
			                       std::vector<std::vector<double>> &A,
			                       std::vector<std::vector<double>> &B,
			                       std::vector<std::vector<double>> &P_t,
			                       std::vector<std::vector<double>> &V,
			                       std::vector<std::vector<double>> &M,
			                       std::vector<double> &x_predicted,
			                       std::vector<std::vector<double>> &P_predicted) {	
	Eigen::VectorXd x_e(x.size());
	Eigen::VectorXd u_e(u.size());
	
	for (size_t i = 0; i < x.size(); i++) {
		x_e[i] = x[i];		
	}
	
	for (size_t i = 0; i < u.size(); i++) {
		u_e[i] = u[i];		
	}
	
	Eigen::MatrixXd A_e(A.size(), A[0].size());
	for (size_t i = 0; i < A.size(); i++) {
		for (size_t j = 0; j < A[0].size(); j++) {
			A_e(i, j) = A[i][j];
		}
	}
	
	Eigen::MatrixXd B_e(B.size(), B[0].size());
	for (size_t i = 0; i < B.size(); i++) {
		for (size_t j = 0; j < B[0].size(); j++) {
			B_e(i, j) = B[i][j];
		}
	}
	
	Eigen::MatrixXd Pt_e(P_t.size(), P_t[0].size());
	for (size_t i = 0; i < P_t.size(); i++) {
		for (size_t j = 0; j < P_t[0].size(); j++) {
			Pt_e(i, j) = P_t[i][j];
		}
	}
	
	Eigen::MatrixXd V_e(V.size(), V[0].size());
	for (size_t i = 0; i < V.size(); i++) {
		for (size_t j = 0; j < V[0].size(); j++) {
			V_e(i, j) = V[i][j];
		}
	}
	
	Eigen::MatrixXd M_e(M.size(), M[0].size());
	for (size_t i = 0; i < M.size(); i++) {
		for (size_t j = 0; j < M[0].size(); j++) {
			M_e(i, j) = M[i][j];
		}
	}
	
	Eigen::VectorXd x_predict(x.size());
	Eigen::MatrixXd P_predict(P_t.size(), P_t[0].size());
	kalmanPredict(x_e, u_e, A_e, B_e, Pt_e, V_e, M_e, x_predict, P_predict);	
}

void KalmanFilter::kalmanUpdate(Eigen::VectorXd &x_predicted,
			                    Eigen::VectorXd &z,
			                    Eigen::MatrixXd &H,
			                    Eigen::MatrixXd &predictedCovariance,
			                    Eigen::MatrixXd &W,
			                    Eigen::MatrixXd &N,
			                    Eigen::VectorXd &x_estimated,
			                    Eigen::MatrixXd &estimatedCovariance) {
	const int size = x_predicted.rows();
	Eigen::MatrixXd kalmanGain(size, size);
	computeKalmanGain(H, predictedCovariance, W, N, kalmanGain);	
	computeStateEstimate(x_predicted, z, H, kalmanGain, x_estimated);
	computeEstimatedCovariance(kalmanGain, H, predictedCovariance, estimatedCovariance);
}

void KalmanFilter::kalmanUpdatePy(std::vector<double> &x,
			                      std::vector<double> &z,
			                      std::vector<std::vector<double>> &H,
			                      std::vector<std::vector<double>> &predictedCovariance,
			                      std::vector<std::vector<double>> &W,
			                      std::vector<std::vector<double>> &N,
			                      std::vector<double> &x_estimated,
			                      std::vector<std::vector<double>> &estimatedCovariance) {
	Eigen::VectorXd x_e(x.size());
	Eigen::VectorXd z_e(z.size());
		
	for (size_t i = 0; i < x.size(); i++) {
		x_e[i] = x[i];		
	}
		
	for (size_t i = 0; i < z.size(); i++) {
		z_e[i] = z[i];		
	}	
	
	Eigen::MatrixXd H_e(H.size(), H[0].size());
	for (size_t i = 0; i < H.size(); i++) {
		for (size_t j = 0; j < H[0].size(); j++) {
			H_e(i, j) = H[i][j];
		}
	}
	
	Eigen::MatrixXd P_e(predictedCovariance.size(), predictedCovariance[0].size());
	for (size_t i = 0; i < predictedCovariance.size(); i++) {
		for (size_t j = 0; j < predictedCovariance[0].size(); j++) {
			P_e(i, j) = predictedCovariance[i][j];
		}
	}
	
	Eigen::MatrixXd W_e(W.size(), W[0].size());
	for (size_t i = 0; i < W.size(); i++) {
		for (size_t j = 0; j < W[0].size(); j++) {
			W_e(i, j) = W[i][j];
		}
	}
	
	Eigen::MatrixXd N_e(N.size(), N[0].size());
	for (size_t i = 0; i < N.size(); i++) {
		for (size_t j = 0; j < N[0].size(); j++) {
			N_e(i, j) = N[i][j];
		}
	}
	
	Eigen::VectorXd x_estimate(x.size());
	Eigen::MatrixXd P_estimate(predictedCovariance.size(), predictedCovariance[0].size());
	kalmanUpdate(x_e, z_e, H_e, P_e, W_e, N_e, x_estimate, P_estimate);
	
}

void KalmanFilter::computePredictedCovariance(Eigen::MatrixXd &A,
			                                  Eigen::MatrixXd &P_t,
			                                  Eigen::MatrixXd &V,
			                                  Eigen::MatrixXd &M,
			                                  Eigen::MatrixXd &predictedCovariance) {
	predictedCovariance = A * (P_t * A.transpose()) + (V * M) * V.transpose();
}

void KalmanFilter::computeKalmanGain(Eigen::MatrixXd &H, 
			                         Eigen::MatrixXd &predictedCovariance,
			                         Eigen::MatrixXd &W, 
			                         Eigen::MatrixXd &N,
			                         Eigen::MatrixXd &kalmanGain) {
	Eigen::MatrixXd H_transpose = H.transpose();
	Eigen::MatrixXd res_temp = H * (predictedCovariance * H_transpose);
	Eigen::MatrixXd res_temp_inv = res_temp.inverse();
	kalmanGain = predictedCovariance * (H_transpose * (H * (predictedCovariance * H_transpose) + W * (N * W.transpose())).inverse());	
}

void KalmanFilter::computeStateEstimate(Eigen::VectorXd &x_predicted,
			                            Eigen::VectorXd &z,
			                            Eigen::MatrixXd &H,
			                            Eigen::MatrixXd &kalmanGain,
			                            Eigen::VectorXd &stateEstimate) {	
    stateEstimate = x_predicted + kalmanGain * (z - H * x_predicted);
}

void KalmanFilter::computeEstimatedCovariance(Eigen::MatrixXd &kalmanGain,
			                                  Eigen::MatrixXd &H,
			                                  Eigen::MatrixXd &predictedCovariance,
			                                  Eigen::MatrixXd &estimatedCovariance) {	
    Eigen::MatrixXd KtH = kalmanGain * H;
    const int size = KtH.rows();
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(size, size);
    estimatedCovariance = (I - KtH) * predictedCovariance;
}

void KalmanFilter::computeLGains(std::vector<Eigen::MatrixXd> &A, 
		                         std::vector<Eigen::MatrixXd> &B, 
		                         Eigen::MatrixXd &C, 
		                         Eigen::MatrixXd &D,
		                         std::vector<Eigen::MatrixXd> &gains) {
	Eigen::MatrixXd S = C;
	std::reverse(A.begin(), A.end());
	std::reverse(B.begin(), B.end());
	for (size_t i = 0; i < A.size(); i++) {		
		Eigen::MatrixXd A_tr = A[i].transpose();
		Eigen::MatrixXd B_tr = B[i].transpose();
		Eigen::MatrixXd L = -(B_tr * S * B[i] + D).inverse() * B_tr * S * A[i];
		gains.push_back(L);
		S = C + A_tr * S * A[i] + A_tr * S * B[i] * L;
	}
	
	std::reverse(gains.begin(), gains.end());	
}

BOOST_PYTHON_MODULE(libKalmanFilter) { 
	using namespace boost::python;
	
	boost::python::type_info info= boost::python::type_id<std::vector<double>>();
	const boost::python::converter::registration* reg_double = boost::python::converter::registry::query(info);
	if (reg_double == NULL || (*reg_double).m_to_python == NULL)  {
	    class_<std::vector<double> > ("v_double")
	    	.def(vector_indexing_suite<std::vector<double> >());
	}
	
	info = boost::python::type_id<std::vector<std::vector<double>>>();
	const boost::python::converter::registration* reg_v2double = boost::python::converter::registry::query(info);
	if (reg_v2double == NULL || (*reg_v2double).m_to_python == NULL)  {  
	    class_<std::vector<std::vector<double> > > ("v2_double")
	    	.def(vector_indexing_suite<std::vector<std::vector<double> > >());
	}
	
	class_<KalmanFilter, boost::shared_ptr<KalmanFilter>>("KalmanFilter", init<>())
			.def("kalmanPredict", &KalmanFilter::kalmanPredictPy)
			.def("kalmanUpdate", &KalmanFilter::kalmanUpdatePy);
}

}