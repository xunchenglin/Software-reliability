#include<iostream>
#include<vector>
#include<math.h>
#include<cmath>
using namespace std;
#define jingdu 0.0001
#define fanwei 0.001
typedef struct error
{
	int i;
	int sum_time;
}error;
//数据
/*0 0 1 9 2 21 3 32 4 36 5 43 6 45 7 50 8 58 9 63 10 70 11 71 12 77 13 78 14 87 
15 91 16 92 17 95 18 98 19 104 20 105 21 116 22 149 23 156 24 247 25 249 26 250 27 337 28 384 -1*/
int main()
{
	int i;
	error temp_error;//临时
	temp_error.i = 0;
	temp_error.sum_time = 0;
	vector<error>error_list;//存所有错误的信息
	double  b=0.0, a=0.0,sum_tk=0;//预算值
	//输入部分
	i = 1;
	cout << "请开始输入错误数据,输-1结束：" << endl;
	while (temp_error.i != -1)//输入
	{
		cin >> temp_error.i;
		if (temp_error.i == -1) {
			break;
		}
		cin >> temp_error.sum_time;
		sum_tk += temp_error.sum_time;
		i++;
		error_list.push_back(temp_error);//放进容器
	}
	const int n = error_list.size() - 1;//实际错误总数
	//计算部分
	a = n - n / 15;//先猜一个a
	double l,r,tn;
	tn = error_list.back().sum_time;
	while(1)
	{
		l = (tn * n) / (sum_tk + tn * (a-n) );
		r = log(a / (a - n));
		//cout << "l:" << l << "  r:" << r << endl;
		if (fabs(l - r) <= fanwei)//tnb和ln(a/(a-n))的差的绝对值小的话
		{
			b = l / tn;
			break;
		}
		a += jingdu;
	}
	//输出环节
	cout << "实际错误个数：" << n << endl;
	cout << "预测a：" << a << endl;
	cout << "预测比例系数b：" << b << endl;
	return 0;
}
