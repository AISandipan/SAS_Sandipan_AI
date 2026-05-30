
/* =====================================
   Insurance Risk & Claims Dashboard SAS
   ===================================== */

/* Import CSV */
PROC IMPORT DATAFILE="insurance_data.csv"
OUT=insurance
DBMS=CSV
REPLACE;
GETNAMES=YES;
RUN;

/* KPI Summary */
PROC SQL;
SELECT 
COUNT(DISTINCT PolicyID) AS TotalPolicies,
SUM(ClaimAmount) AS TotalClaimAmount,
AVG(ClaimFrequency) AS AvgClaimFreq,
AVG(ClaimAmount) AS AvgClaimAmount
FROM insurance;
QUIT;

/* Gender Counts */
PROC FREQ DATA=insurance;
TABLES Gender;
RUN;

/* Car Use Claims */
PROC SQL;
CREATE TABLE car_use AS
SELECT CarUse,
SUM(ClaimAmount) AS ClaimAmount
FROM insurance
GROUP BY CarUse;
QUIT;

/* Car Make Claims */
PROC SQL;
CREATE TABLE car_make AS
SELECT CarMake,
SUM(ClaimAmount) AS ClaimAmount
FROM insurance
GROUP BY CarMake
ORDER BY ClaimAmount DESC;
QUIT;

/* Coverage Zone */
PROC SQL;
CREATE TABLE coverage_zone AS
SELECT CoverageZone,
SUM(ClaimAmount) AS ClaimAmount
FROM insurance
GROUP BY CoverageZone;
QUIT;

/* Age Groups */
PROC SQL;
CREATE TABLE age_grp AS
SELECT AgeGroup,
SUM(ClaimAmount) AS ClaimAmount
FROM insurance
GROUP BY AgeGroup;
QUIT;

/* Car Year Trend */
PROC SQL;
CREATE TABLE car_year AS
SELECT CarYear,
SUM(ClaimAmount) AS ClaimAmount
FROM insurance
GROUP BY CarYear;
QUIT;

/* Education x Marital Matrix */
PROC TABULATE DATA=insurance;
CLASS Education MaritalStatus;
VAR ClaimAmount;

TABLE Education,
MaritalStatus*SUM*ClaimAmount;
RUN;
