#include<iostream>
#include<vector>
#include<math.h>
#include<cmath>
using namespace std;
#define jingdu 0.00001
#define fanwei 0.0001
typedef struct error
{
	int i;
	int time;
}error;

double summ(int n,double N,double xxii,double & sum_fen)//�����ֵ
{
	sum_fen = 0;
	for (int i = 1; i <= n; i++)
	{
		sum_fen += (double)1/ (N - i + 1);
	}
	return sum_fen-n/(N-xxii);
}
//���ݣ�
/* 1 9 2 12 3 11 4 4 5 7 6 2 7 5 8 8 9 5 10 7 11 1 12 6 13 1 14 9 15 4 16 1 17 3 
18 8 19 6 20 1 21 1 22 33 23 7 24 91 25 2 26 1 27 87 28 47 29 12 30 9 31 135 32 258 33 16 34 35 0*/
int main()
{
	int i ;
	error temp_error;//��ʱ
	temp_error.i = 0;
	temp_error.time = 0;
	vector<error>error_list;//�����д������Ϣ
	double sum_xi = 0, sum_ixi = 0,guessF = 0;
	double s, tt, N,sum_fen,j;
	//���벿��
	i = 1;
	cout << "�뿪ʼ�����������,��0������" << endl;
	while (1)//����
	{
		cin >> temp_error.i;
		if (temp_error.i == 0){
			break;
		}
		cin >> temp_error.time;
		sum_xi += temp_error.time;//xi�ܺ�//��ǰ����Ҫ�õ�ֵ
		sum_ixi += temp_error.time * (i-1); //��i-1��xi�ܺ�//��ǰ����Ҫ�õ�ֵ
		i++;
		error_list.push_back(temp_error);//�Ž�����
	}
	double xxii = sum_ixi / sum_xi;//��ǰ����Ҫ�õ�ֵ
	const int n = error_list.size();//ʵ�ʴ�������
	//���㲿��
	N = n-n/15;//�Ȳ�NΪ����n
	double sn = summ(n, N, xxii, sum_fen);;
	while(guessF==0)//faiΪ�Ǻ�
	{
		sn = summ(n, N, xxii, sum_fen);
		if (fabs(sn) <= fanwei )//��ֵ����ֵС�ڵ���
		{
			guessF = (float)n / (N * sum_xi - sum_ixi);//ȷ��fai,Ҳ��Ϊ�Ǻ�
			break;//����
		}
		else if(fabs(sn) > fanwei)//��ֵ����ֵ
		{
			while (sn > 0 && fabs(sn) > fanwei)
			{
				N += jingdu;
				sn = summ(n, N, xxii, sum_fen);
			}
			while (sn < 0 && fabs(sn) > fanwei)
			{
				N -= jingdu;
				sn = summ(n, N, xxii, sum_fen);
			}
			guessF = (float)n / (N * sum_xi - sum_ixi);//ȷ��fai,Ҳ��Ϊ�Ǻ�
			break;//����
		}
	}
	//�������
	cout << "Ԥ����������" << N <<endl;
	cout << "ʵ�ʴ��������" << error_list.size() << endl;
	cout << "Ԥ�����ϵ����" << guessF <<endl;
	return 0;
}