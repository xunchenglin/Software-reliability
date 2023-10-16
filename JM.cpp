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

double summ(int n,double N,double xxii,double & sum_fen)//计算差值
{
	sum_fen = 0;
	for (int i = 1; i <= n; i++)
	{
		sum_fen += (double)1/ (N - i + 1);
	}
	return sum_fen-n/(N-xxii);
}
//数据：
/* 1 9 2 12 3 11 4 4 5 7 6 2 7 5 8 8 9 5 10 7 11 1 12 6 13 1 14 9 15 4 16 1 17 3 
18 8 19 6 20 1 21 1 22 33 23 7 24 91 25 2 26 1 27 87 28 47 29 12 30 9 31 135 32 258 33 16 34 35 0*/
int main()
{
	int i ;
	error temp_error;//临时
	temp_error.i = 0;
	temp_error.time = 0;
	vector<error>error_list;//存所有错误的信息
	double sum_xi = 0, sum_ixi = 0,guessF = 0;
	double s, tt, N,sum_fen,j;
	//输入部分
	i = 1;
	cout << "请开始输入错误数据,输0结束：" << endl;
	while (1)//输入
	{
		cin >> temp_error.i;
		if (temp_error.i == 0){
			break;
		}
		cin >> temp_error.time;
		sum_xi += temp_error.time;//xi总和//提前计算要用的值
		sum_ixi += temp_error.time * (i-1); //（i-1）xi总和//提前计算要用的值
		i++;
		error_list.push_back(temp_error);//放进容器
	}
	double xxii = sum_ixi / sum_xi;//提前计算要用的值
	const int n = error_list.size();//实际错误总数
	//计算部分
	N = n-n/15;//先猜N为倍的n
	double sn = summ(n, N, xxii, sum_fen);;
	while(guessF==0)//fai为记号
	{
		sn = summ(n, N, xxii, sum_fen);
		if (fabs(sn) <= fanwei )//差值绝对值小于等于
		{
			guessF = (float)n / (N * sum_xi - sum_ixi);//确定fai,也作为记号
			break;//跳出
		}
		else if(fabs(sn) > fanwei)//差值绝对值
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
			guessF = (float)n / (N * sum_xi - sum_ixi);//确定fai,也作为记号
			break;//跳出
		}
	}
	//输出环节
	cout << "预测错误个数：" << N <<endl;
	cout << "实际错误个数：" << error_list.size() << endl;
	cout << "预测比例系数：" << guessF <<endl;
	return 0;
}