/*=========================================================
1. Customer Segmentation by Balance
=========================================================*/
%macro customer_segment;
proc sql;
select CustomerID,
       Name,
       Balance,
       case
          when Balance >= 100000 then 'Premium'
          when Balance >= 50000 then 'Gold'
          else 'Regular'
       end as Segment
from WORK.customer_sas_new;
quit;
%mend;
%customer_segment;


/*=========================================================
2. Top Profitable Customers
=========================================================*/
%macro top_profit(n);
proc sql outobs=&n;
select *
from WORK.customer_sas_new
order by Profit desc;
quit;
%mend;
%top_profit(10);


/*=========================================================
3. Service Wise Profit Analysis
=========================================================*/
%macro service_profit;
proc sql;
select Service,
       sum(Profit) as Total_Profit,
       avg(Profit) as Avg_Profit
from WORK.customer_sas_new
group by Service;
quit;
%mend;
%service_profit;


/*=========================================================
4. Gender Wise Sales Analysis
=========================================================*/
%macro gender_sales;
proc sql;
select Gender,
       count(*) as Customers,
       sum(Sales) as Total_Sales
from WORK.customer_sas_new
group by Gender;
quit;
%mend;
%gender_sales;


/*=========================================================
5. High Value Customers
=========================================================*/
%macro high_value(amount);
proc sql;
select *
from WORK.customer_sas_new
where Sales >= &amount;
quit;
%mend;
%high_value(100000);


/*=========================================================
6. Account Type Performance
=========================================================*/
%macro account_performance;
proc sql;
select AccountType,
       count(*) as Customers,
       sum(Sales) as Sales,
       sum(Profit) as Profit
from WORK.customer_sas_new
group by AccountType;
quit;
%mend;
%account_performance;


/*=========================================================
7. Age Group Analysis
=========================================================*/
%macro age_group;
proc sql;
select
case
 when Age < 30 then 'Young'
 when Age between 30 and 40 then 'Middle'
 else 'Senior'
end as Age_Group,
count(*) as Customers
from WORK.customer_sas_new
group by calculated Age_Group;
quit;
%mend;
%age_group;


/*=========================================================
8. Customer Ranking by Sales
=========================================================*/
%macro sales_rank;
proc rank data=WORK.customer_sas_new
          out=ranked_customer
          descending;
var Sales;
ranks Sales_Rank;
run;

proc print data=ranked_customer;
run;
%mend;
%sales_rank;


/*=========================================================
9. Profit Margin Analysis
=========================================================*/
%macro profit_margin;
proc sql;
select *,
       round((Profit/Sales)*100,2) as Profit_Margin
from WORK.customer_sas_new;
quit;
%mend;
%profit_margin;


/*=========================================================
10. City Performance Report
=========================================================*/
%macro city_performance;
proc sql;
select City,
       sum(Sales) as Sales,
       sum(Profit) as Profit
from WORK.customer_sas_new
group by City
order by Sales desc;
quit;
%mend;
%city_performance;


/*=========================================================
11. Loan Customer Report
=========================================================*/
%macro loan_customers;
proc sql;
select *
from WORK.customer_sas_new
where Service='Loan';
quit;
%mend;
%loan_customers;


/*=========================================================
12. Premium Customer List
=========================================================*/
%macro premium_customer;
proc sql;
select *
from WORK.customer_sas_new
where Balance >= 100000;
quit;
%mend;
%premium_customer;


/*=========================================================
13. Dynamic Service Report
=========================================================*/
%macro service_report(service);
proc sql;
select *
from WORK.customer_sas_new
where Service="&service";
quit;
%mend;
%service_report(Investment);


/*=========================================================
14. Customer KPI Dashboard
=========================================================*/
%macro kpi_dashboard;
proc sql;
select
count(*) as Customers,
sum(Sales) as Total_Sales,
sum(Profit) as Total_Profit,
avg(Balance) as Avg_Balance
from WORK.customer_sas_new;
quit;
%mend;
%kpi_dashboard;


/*=========================================================
15. Export Summary Report
=========================================================*/
%macro export_report;
proc summary data=WORK.customer_sas_new;
var Sales Profit Balance;
output out=summary_report
sum=;
run;

proc export data=summary_report
outfile="C:\Temp\Customer_Report.csv"
dbms=csv
replace;
run;
%mend;
%export_report;