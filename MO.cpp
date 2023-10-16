#include<iostream>
#include<vector>
#include<math.h>
#include<cmath>
using namespace std;
#define jingdu 0.1
#define fanwei 0.1
typedef struct error
{
	int i;
	int sum_time;
}error;

double sum(double beta, vector<error>error_list)
{
	double sum_fen = 0;
	for (int i = 0; i < error_list.size(); i++)
	{
		sum_fen += error_list[i].sum_time / (beta * error_list[i].sum_time + 1);
	}
	return sum_fen;
}
//����
/*0 0 1 9 2 21 3 32 4 36 5 43 6 45 7 50 8 58 9 63 10 70 11 71 12 77 13 78 14 87
15 91 16 92 17 95 18 98 19 104 20 105 21 116 22 149 23 156 24 247 25 249 26 250 27 337 28 384 -1*/
int main()
{
	int i;
	error temp_error;//��ʱ
	temp_error.i = 0;
	temp_error.sum_time = 0;
	vector<error>error_list;//�����д������Ϣ
	double theta,fai,beta,sum_fen=0,r,d;
	//���벿��
	i = 1;
	cout << "�뿪ʼ�����������,��-1������" << endl;
	while (temp_error.i != -1)//����
	{
		cin >> temp_error.i;
		if (temp_error.i == -1) {
			break;
		}
		cin >> temp_error.sum_time;
		i++;
		error_list.push_back(temp_error);//�Ž�����
	}
	const int n = error_list.size() - 1;//ʵ�ʴ�������
	double tn = error_list.back().sum_time;
	//���㲿��
	beta = 0;
	while (1)
	{
		beta += jingdu;
		sum_fen = sum(beta, error_list);
		d= (beta * tn + 1) * log(beta * tn + 1);
		r = n * tn / d;
		if (fabs(n/beta-sum_fen-r) <= fanwei)
		{
			theta = (log(beta * tn + 1)) / n;
			fai = beta / theta;
			break;
		}
	}

	//�������
	cout << "Ԥ��betaΪ��" << beta << endl;
	cout << "Ԥ��thetaΪ��" << theta << endl;
	cout << "Ԥ��faiΪ��" << fai << endl;
	return 0;
}