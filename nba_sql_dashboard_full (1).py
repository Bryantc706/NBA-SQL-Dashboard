# SQL Group Project
# NBA Dashboard: Isaiah Hawbaker, Simon Michel, Thomas Gaza, Bryant Connolly
# streamlit run nba_sql_dashboard_full.py

# import modules/libraries
import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# set the page configuration
st.set_page_config(layout = "wide")

### Database Connection ###
# set the path the database file
data_path = "/Users/thomasgaza/Desktop/SQL Project/archive/nba.sqlite"
# connect to the SQLite database
connection = sqlite3.connect(data_path)
# create a cursor object to execute SQL query
cursor = connection.cursor()

### SQL Queries ###
# Isaiah Hawbaker
# Q1 (Isaiah)
query_1 = """SELECT 
    CASE 
        WHEN TRIM(LOWER("position")) IN ('center', 'center-forward') THEN 'Center'
        WHEN TRIM(LOWER("position")) IN ('forward', 'forward-guard', 'forward-center') THEN 'Forward'
        ELSE 'Guard'
    END AS "NBA_Position",
    ROUND(AVG(season_exp), 2) AS "Seasons Played on Average"
FROM common_player_info cpi 
GROUP BY "NBA_Position"
ORDER BY "Seasons Played on Average" DESC"""

# execute the query 
cursor.execute(query_1)

# fetch all results from the execute query
results_1 = cursor.fetchall()
results_1 = pd.DataFrame(results_1)
# name the columns
results_1.columns = ['NBA_Position', 'Seasons Played on Average']

# Q2 (Isaiah)
query_2 = """SELECT
    CASE 
        WHEN CAST(from_year AS INT) BETWEEN 1940 AND 1949 THEN '1940s'
        WHEN CAST(from_year AS INT) BETWEEN 1950 AND 1959 THEN '1950s'
        WHEN CAST(from_year AS INT) BETWEEN 1960 AND 1969 THEN '1960s'
        WHEN CAST(from_year AS INT) BETWEEN 1970 AND 1979 THEN '1970s'
        WHEN CAST(from_year AS INT) BETWEEN 1980 AND 1989 THEN '1980s'
        WHEN CAST(from_year AS INT) BETWEEN 1990 AND 1999 THEN '1990s'
        WHEN CAST(from_year AS INT) BETWEEN 2000 AND 2009 THEN '2000s'
       	WHEN CAST(from_year AS INT) BETWEEN 2010 AND 2019 THEN '2010s'
        ELSE '2020s'
    END AS Decade,
    COUNT(greatest_75_flag) AS "NBA 75th Anniversary Team Players"
FROM common_player_info cpi 
WHERE greatest_75_flag = 'Y'
GROUP BY Decade
ORDER BY "NBA 75th Anniversary Team Players" DESC"""

# execute the query 
cursor.execute(query_2)

# fetch all results from the execute query
results_2 = cursor.fetchall()
results_2 = pd.DataFrame(results_2)
# name the columns
results_2.columns = ['Decade', 'NBA 75th Anniversary Team Players']

# Simon Michel
# Q1 (Simon)
query_3 = """SELECT STRFTIME('%Y', game_date) AS 'Year', 
		                    AVG(fg3m_home) AS '3PM Home',
		                    AVG(fg3a_home) AS '3PA Home',
		                    AVG(fg3_pct_home) AS '3P% Home',
		                    AVG(fg3m_away) AS '3PM Away',
		                    AVG(fg3a_away) AS '3PA Away',
		                    AVG(fg3_pct_away) AS '3P% Away'
                        FROM game as g
                        WHERE STRFTIME('%Y', game_date) > '2009'
		                    and wl_home = 'W' 
		                    and wl_away = 'L'
                        GROUP BY STRFTIME('%Y', game_date)
                        ORDER BY STRFTIME('%Y', game_date) DESC"""

# Execute the query
cursor.execute(query_3)
# Fetch results and define column names
results_3 = cursor.fetchall()
columns = ["Year", "Winner AVG 3PM PG", "Winner AVG 3PA PG", "Winner AVG 3P% PG",
           "Loser AVG 3PM PG", "Loser AVG 3PA PG", "Loser AVG 3P% PG"]
# Create a DataFrame
results_3 = pd.DataFrame(results_3, columns=columns)

# Q2 (Simon)
query_4 = """SELECT STRFTIME('%Y', g.game_date) AS 'Year', 
		                    AVG(g.reb_home) AS 'Rebounds Home',
		                    AVG(os.pts_paint_home) AS 'Points in the Paint Home',
		                    AVG(os.pts_2nd_chance_home) AS '2nd Chance Points Home',
		                    AVG(g.reb_away) AS 'Rebounds Away',
		                    AVG(os.pts_paint_away) AS 'Points in the Paint Away',
		                    AVG(os.pts_2nd_chance_away) AS '2nd Chance Points Away'
                        FROM game as g
                        INNER JOIN other_stats as os
                        ON g.game_id = os.game_id
                        WHERE STRFTIME('%Y', g.game_date) > '2009'
		                    and wl_home = 'W'
		                    and wl_away = 'L'
                        GROUP BY STRFTIME('%Y', g.game_date)
                        ORDER BY STRFTIME('%Y', g.game_date) DESC"""

cursor.execute(query_4)
# Fetch results and define column names
results_4 = cursor.fetchall()
columns = ["Year", "Winner AVG REB PG", "Winner AVG PTSPAINT PG", "Winner AVG 2NDCHANCEPTS PG",
           "Loser AVG REB PG", "Loser AVG PTSPAINT PG", "Loser AVG 2NDCHANCEPTS PG"]
# Create a DataFrame
results_4 = pd.DataFrame(results_4, columns=columns)
# Set the Year column as the index
results_4.set_index("Year", inplace=True)

# Thomas Gaza
# Q1 (Thomas)
query_5 = """WITH RankedPositions AS (
    SELECT 
        CASE 
            WHEN from_year BETWEEN 1946 AND 1948 THEN 'The BAA Era'
            WHEN from_year BETWEEN 1949 AND 1966 THEN 'BAA-NBL Merger'
            WHEN from_year BETWEEN 1967 AND 1975 THEN 'The Expansion Era'
            WHEN from_year BETWEEN 1976 AND 1977 THEN 'ABA-NBA Merger'
            WHEN from_year BETWEEN 1978 AND 1988 THEN 'The Post Merger Era'
            WHEN from_year BETWEEN 1989 AND 2004 THEN 'The Modern Expansion Era'
            WHEN from_year BETWEEN 2005 AND 2024 THEN 'The Realignment Era'
        END AS decade,
        position,
        COUNT(*) AS position_count,
        ROW_NUMBER() OVER (PARTITION BY 
            CASE 
	            WHEN from_year BETWEEN 1946 AND 1948 THEN 'The BAA Era'
	            WHEN from_year BETWEEN 1949 AND 1966 THEN 'BAA-NBL Merger'
	            WHEN from_year BETWEEN 1967 AND 1975 THEN 'The Expansion Era'
	            WHEN from_year BETWEEN 1976 AND 1977 THEN 'ABA-NBA Merger'
	            WHEN from_year BETWEEN 1978 AND 1988 THEN 'The Post Merger Era'
	            WHEN from_year BETWEEN 1989 AND 2004 THEN 'The Modern Expansion Era'
	            WHEN from_year BETWEEN 2005 AND 2024 THEN 'The Realignment Era'
            END
        ORDER BY COUNT(*) DESC) AS row_num
    FROM common_player_info
    GROUP BY decade, position
)
SELECT decade, position, position_count
FROM RankedPositions
WHERE row_num = 1
ORDER BY 
    CASE 
        WHEN decade = 'The BAA Era' THEN 1
        WHEN decade = 'BAA-NBL Merger' THEN 2
        WHEN decade = 'The Expansion Era' THEN 3
        WHEN decade = 'ABA-NBA Merger' THEN 4
        WHEN decade = 'The Post Merger Era' THEN 5
        WHEN decade = 'The Modern Expansion Era' THEN 6
        WHEN decade = 'The Realignment Era' THEN 7
    END"""

# execute the query 
cursor.execute(query_5)

# fetch all results from the execute query
results_5 = cursor.fetchall()
results_5 = pd.DataFrame(results_5)
# name the columns
results_5.columns = ['Decade', 'Position', 'Count of Position Drafted']

# Q2 (Thomas)
query_6 = """SELECT 
    g.team_name_home, 
    COUNT(*) AS home_win_count,
    r.revenue / COUNT(*) as rev_per_home_win
FROM 
    game as g
JOIN 
    team_revenue as r 
ON 
	g.team_name_home = r.team_name
WHERE 
    g.wl_home = 'W' AND g.game_date BETWEEN '2022-10-18 00:00:00' AND '2023-04-09 23:59:59'
GROUP BY
    g.team_name_home, r.revenue
ORDER BY 
    rev_per_home_win DESC
LIMIT 10"""

# execute the query 
cursor.execute(query_6)

# fetch all results from the execute query
results_6 = cursor.fetchall()
results_6 = pd.DataFrame(results_6)
# name the columns
results_6.columns = ['Team Name', 'Home Wins', 'Revenue per Home Win (in Millions USD)']

# Bryant Connolly
# Q1 (Bryant)
query_7 = """
SELECT 
    gi.game_id,
    gi.game_date,
    gi.attendance,
    g.team_abbreviation_home AS home_team,
    g.team_abbreviation_away AS away_team
FROM 
    game_info gi
JOIN 
    game g ON gi.game_id = g.game_id
WHERE 
    (g.team_abbreviation_home = 'WST' AND g.team_abbreviation_away = 'EST')
    OR 
    (g.team_abbreviation_home = 'EST' AND g.team_abbreviation_away = 'WST')
ORDER BY 
    gi.attendance DESC;
"""

# Connect to SQLite database and execute the query
try:
    connection = sqlite3.connect(data_path)
    data_1 = pd.read_sql_query(query_7, connection)
    # st.subheader("WST vs EST Attendance Data")
    # st.dataframe(data)
except Exception as e:
    st.error(f"An error occurred: {e}")

# Clean up the data for visualization
data_1["game_date"] = pd.to_datetime(data_1["game_date"])  # Ensure proper datetime format
data_1["game_year"] = data_1["game_date"].dt.year  # Extract the year for visualization

# Filter data to include only years >= 2000
data_1 = data_1[data_1["game_year"] >= 2000]

# Q2 (Bryant)
query_8 = """
SELECT 
    dcs.player_id,
    dcs.player_name, 
    dcs.position,
    dcs.height_w_shoes,
    dcs.weight,
    dcs.wingspan,
    dcs.standing_vertical_leap,
    dcs.max_vertical_leap,
    dcs.lane_agility_time,
    dcs.modified_lane_agility_time,
    dcs.three_quarter_sprint,
    dcs.bench_press,
    dcs.spot_fifteen_corner_left,
    dcs.spot_fifteen_break_left,
    dcs.spot_nba_corner_right,
    dcs.off_drib_college_top_key,
    dcs.hand_length,
    dcs.spot_nba_break_right,
    dcs.hand_width,
    dcs.spot_nba_top_key, 
    dcs.body_fat_pct, 
    dh.overall_pick AS draft_position,
    dh.round_number AS draft_round,
    dh.season AS draft_year
FROM 
    draft_combine_stats AS dcs
JOIN 
    draft_history AS dh
ON 
    dcs.player_id = dh.person_id;
"""
connection = sqlite3.connect(data_path)
data = pd.read_sql_query(query_8, connection)
connection.close()

# Drop irrelevant columns
data_cleaned = data.drop(columns=[
    "player_id", "player_name", "draft_round", 
    "spot_fifteen_corner_left", "spot_fifteen_break_left", "draft_year",
    "off_drib_college_top_key"
])

# Handle missing values: Convert to numeric and replace NaN with 0
for col in data_cleaned.columns:
    if data_cleaned[col].dtype == "object":
        data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')

data_cleaned = data_cleaned.fillna(0)

# Define features (X) and target (y)
X = data_cleaned.drop(columns=["draft_position"])
y = data_cleaned["draft_position"]

# Encode categorical columns
X = pd.get_dummies(X, columns=["position", "spot_nba_corner_right", 
                               "spot_nba_break_right", "spot_nba_top_key"], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

### Close Connection ###
connection.close()

### Dashboard Design ###
# dashboard title
st.title("NBA Group Project Dashboard :basketball:")
st.caption('Isaiah Hawbaker, Simon Michel, Thomas Gaza, Bryant Connolly')
st.divider()

# Isaiah Dashboard Elements
# Q1
st.header("Q1: Career Longevity")
st.write("**What position in the NBA has the best career longevity?**")
st.write("*We’re in free agency and there are three players with a similar skillset at the ripe age of 30, all three of them want a new 4-year contract, but we can only afford to give that type of contract to one of them. One plays guard, another plays forward, and the last plays center. With them up there in age and close to retirement, we want to give the 4-year contract to the position with the best longevity.*")
st.bar_chart(results_1,
             x = 'NBA_Position',
             y = 'Seasons Played on Average',
             color = 'NBA_Position',
             horizontal = True)
st.write("**Business Statement**: *We will sign the center, as data indicates that centers generally exhibit the best career longevity in the NBA. This decision maximizes our investment by ensuring we get the most value from the 4-year contract, considering their potential to perform effectively for a longer period.*")
st.divider()

# Q2
st.header("Q2: Best Decade")
st.write("**What decade of NBA basketball has the most “NBA 75th Anniversary Team” players?**")
st.write("*We have just agreed to a contract with Netflix to film a series titled “The Greatest Era of Basketball” in honor of the NBA 75th Anniversary. This series will display episodes tailored to different players from a specific decade in the NBA. Netflix has asked us to choose the decade with the most players listed on the NBA 75th Anniversary Team as they strongly believe that is the decade that should be showcased in the new series!*")
st.line_chart(results_2,
              x = 'Decade',
              y = 'NBA 75th Anniversary Team Players',
              color = '#89cff0')
st.write("**Business Statement**: *Netflix should center the series 'The Greatest Era of Basketball' around either the 1960s or the 1980s, as these decades boast the highest number of NBA 75th Anniversary Team members. The 1960s featured legendary figures like Wilt Chamberlain, Oscar Robertson, Jerry West, Bill Russell, and Bob Pettit, who defined the early era of modern basketball. Meanwhile, the 1980s brought unparalleled global appeal with stars such as Michael Jordan, Larry Bird, Kareem Abdul-Jabbar, Magic Johnson, and Julius Erving, showcasing the league's transformation into a worldwide phenomenon. Highlighting these decades will captivate audiences by exploring the most influential players and pivotal moments in NBA history.*")
st.divider()

# Simon Dashboard Elements
# Q1
st.header("Q3: Impact of 3-Point Shooting on Wins")
st.write("**In the Modern NBA (from 2010 to present), for winning home teams, is the average for 3 pointers made, 3 pointers attempted, and 3 point percentage greater per season than for losing away teams?**")
st.dataframe(results_3, use_container_width=True)
threes_made_df = results_3[["Winner AVG 3PM PG", "Loser AVG 3PM PG"]]
threes_attempted_df = results_3[["Winner AVG 3PA PG", "Loser AVG 3PA PG"]]
three_point_percentage_df = results_3[["Winner AVG 3P% PG", "Loser AVG 3P% PG"]]
st.subheader("Average Three Pointers Made Per Game by Season")
st.line_chart(threes_made_df)
st.subheader("Average Three Pointers Attempted Per Game by Season")
st.line_chart(threes_attempted_df)
st.subheader("Average Three Point Percentage Per Game by Season")
st.line_chart(three_point_percentage_df)
#st.subheader("Conclusion")
st.write("""
**Business Statement:** *Based on the results in these graphs, it's safe to say that 3-point shooting accuracy has a significant
impact on a team's ability to win in the modern NBA (2010-present). The graphs show that winning isn't based
so much on the number of 3-pointers each team attempts, as average attempts are about the same for winners
and losers, but based much more on the number of 3-pointers made. The winning team made significantly
more 3-pointers per game and had a higher shooting percentage from beyond the arc. The business insight
derived from these findings is that winning in the modern NBA relies on players' ability to take and make
many 3-pointers per game, which means teams should target 3-point sharpshooters in scouting and the draft.
Reliable 3-point shooting is usually associated with point guards, shooting guards, and small forwards,
who tend to be the best 3-point shooters on a team, so it would stand to reason that teams should look to
draft players at these positions who have high 3-point percentage averages.*
""")
st.divider()

# Q2 
st.header("Q4: Impact of Paint Stats on Wins")
st.write("**In the Modern NBA (from 2010 to present), for winning home teams, is the average for rebounds, points in the paint, and 2nd chance points greater per season than for losing away teams?**")
st.dataframe(results_4, use_container_width=True)
rebounds_df = results_4[["Winner AVG REB PG", "Loser AVG REB PG"]]
pts_paint_df = results_4[["Winner AVG PTSPAINT PG", "Loser AVG PTSPAINT PG"]]
second_chance_pts_df = results_4[["Winner AVG 2NDCHANCEPTS PG", "Loser AVG 2NDCHANCEPTS PG"]]
st.subheader("Average Rebounds Per Game by Season")
st.line_chart(rebounds_df)
st.subheader("Average Points in the Paint Per Game by Season")
st.line_chart(pts_paint_df)
st.subheader("Average Second Chance Points Per Game by Season")
st.line_chart(second_chance_pts_df)
# st.subheader("Conclusion")
st.write("""
**Business Statement:** *Based on the results in these graphs, it's safe to assume that rebounding has a significant impact on
winning in the modern NBA (2010-present), with the winning teams having significantly higher rebounds
per game in each season than the losing teams. Rebounding is a statistic typically associated with the
big men on the floor (i.e. power forwards and centers), but any player on the floor can grab rebounds,
and in the modern NBA, it's not always the big men at those positions who lead their teams in rebounding.
For example, Luka Doncic plays point guard on the Dallas Mavericks and leads his team in points, rebounds,
and assists per game despite the Mavericks' strong core of rebounding power forwards and centers. These
results offer an indictment on big men in the modern NBA, as points in the paint and second chance points
(i.e. points from offensive rebounds) do not significantly impact a team's ability to win. The difference
between winning and losing teams' points in the paint and second chance points is marginal. The business
insight that can be derived from these graphs is that teams should not heavily focus on scouting and
drafting power forwards and centers who exclusively play in the paint, as paint statistics don't have
an incredible impact on winning in the modern NBA. Instead, they should focus on scouting more versatile
big men who have a greater shooting range and can contribute in other statistical categories, like
assists and 3-pointers.*
""")
st.divider()

# Thomas Dashboard Elements
# Q1
st.header("Q5: Top Position Drafted Each Era")
st.write("**What is the most popular position drafted in each era of NBA basketball?**")
st.markdown("*The Front Office and Scouting Department is lookng to be one step ahead of the next trend in drafting potential NBA prospects. To understand where the trend is heading, the Scouting Department want to look at the popularity of different positions across differents eras of basketball.*")
st.dataframe(results_5, use_container_width=True)
st.markdown('**Business Statement:** *Over the years, the Draft has shown a cyclical pattern in the preference for Guards and Forwards, with each position experiencing shifts in popularity across different eras. If this trend holds, we can expect a swing back to the Forward position in the near future.*')
st.divider()

# Q2
st.header("Q6: Top 10 NBA Teams Ranked by Revenue per Home Win")
st.write("**Which NBA team generates the highest revenue per home win?**")
st.markdown("*Ownership is looking to boost the team’s stadium revenue as they look to potentially build a new stadium in the near future. One of the most obvious ways to boost fan retention and therefore revenue is winning in front of the team's fanbase. By identifying teams that generate the highest revenue per home win, we can analyze their stadium's business models and implement effective strategies to optimize our own revenue.*")
# horizontal bar chart
st.bar_chart(
    results_6, 
    x = 'Team Name',
    y = ['Revenue per Home Win (in Millions USD)'],
    color = '#ff8c00', # set the color to orange
    horizontal = True)
st.dataframe(results_6, use_container_width=True)
st.markdown('**Business Statement:** *These are the Top 10 NBA teams that generate the most revenue, in millions of USD, for winning games at home. If teams around the league want to boost their stadium revenue for each game won in front of the fans, they should investigate the business model of the top 10 teams and implement those ideas.*')
st.divider()

# Bryant Dashboard Elements
# Q1
st.header("Q7: NBA All Star Game Attendance")
st.write("**For the buiness question that I wanted to answer for this case was that I wanted to examine the past all star games for the NBA to see if there is any trends or outliers to examine and to see how to improve it overtime.**")
# st.dataframe(data)
# Bar Graph Visualization
st.subheader("Attendance by Year (Bar Graph)")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(data_1["game_year"], data_1["attendance"], color="skyblue")
ax.set_xlabel("Year of All Star Game", fontsize=12)
ax.set_ylabel("Attendance of All Star Game", fontsize=12)
ax.set_title("Attendance by Year", fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Scatterplot Visualization
st.subheader("Attendance by Year (Scatterplot)")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.scatter(data_1["game_year"], data_1["attendance"], color="blue", s=44)  # s sets the marker size
ax2.set_xlabel("Year of All Star Game", fontsize=12)
ax2.set_ylabel("Attendance of All Star Game", fontsize=12)
ax2.set_title("Attendance by Year", fontsize=14)
ax2.grid(True, linestyle="--", alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)
st.write("**Business Insight**: *From looking into the scatterplot and the bar graph for NBA All Star Game attendance, I noticed that there was an outlier from 2010 where the amount of people that went to that game versus past years and future years were shown to be significantly higher. From examining the different years for all star games I wanted to examine why specifically in 2010 that there was over 108,713 people in attendance. From doing further research it was shown that that perticular game was held at At&T Stadium which is where the Dallas Cowboys play. Having the NBA All Star Game in a football stadium versus a basketball arena gave the NBA the opportunity to sell more tickets for people to attend the event as it is one of the more notable events that the NBA tries to promote during the season as it brings the best players from the Eastern and Western Conferences. With that, I believe that the NBA should look further into having their all star games being held in a football stadium as it gives more of an opportunity to make more money for the product as it would bring more revenue with more tickets as a football stadium has a much larger capacity versus a basketball arena.*")
st.divider()

# Q2
st.header("Q8: NBA Draft Position Prediction with Linear Regression and Random Forest Classifier Models")
st.write("**Business Question: What are statistics that are looked for by the NBA combine that contribute towards a player being drafted higher versus another or what metrics help a player to be drafted versus another player?**")
st.write("*Reasoning: I want to investigate what are certain metrics that can impact how it can affect a player draft stock or position from rising or dropping in the draft.*")

# Display Evaluation Metrics
st.subheader("Model Evaluation")
# st.divider()
st.write("**Random Forest Results:**")
st.write(f"Mean Squared Error (MSE): {mse_rf:.2f}")
st.write(f"R-squared (R2): {r2_rf:.2f}")
st.divider()
st.write("**Linear Regression Results:**")
st.write(f"Mean Squared Error (MSE): {mse_lr:.2f}")
st.write(f"R-squared (R2): {r2_lr:.2f}")

# Feature Importance Visualization for Random Forest
feature_importances = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.subheader("Feature Importance Visualization (Random Forest)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(feature_importances["Feature"], feature_importances["Importance"], color="skyblue")
ax.set_title("Feature Importance (Random Forest)")
ax.set_xlabel("Features")
ax.set_ylabel("Importance")
ax.tick_params(axis='x', rotation=90)
st.pyplot(fig)

# Linear Regression Coefficients
lr_coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": lr_model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.subheader("Feature Coefficients (Linear Regression)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(lr_coefficients["Feature"], lr_coefficients["Coefficient"], color="lightgreen")
ax.set_title("Feature Coefficients (Linear Regression)")
ax.set_xlabel("Features")
ax.set_ylabel("Coefficient")
ax.tick_params(axis='x', rotation=90)
st.pyplot(fig)

# Residual Plot for Linear Regression
residuals = y_test - y_pred_lr
st.subheader("Residual Plot (Linear Regression)")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_pred_lr, residuals, alpha=0.6, color="orange")
ax.axhline(y=0, color='blue', linestyle='--')
ax.set_title("Residual Plot (Linear Regression)")
ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
st.pyplot(fig)


st.write("**Business Statement:** *Insight of linear regression: From creating a  linear regression and a random forest classifier, I first created a linear regression to see how the metrics would correlate with draft position and the results from it showed that there is a very low R-squared score for it, which contributes that the metrics provided do not correlate or align with the draft position of the player. With the coefficients it shows that a three quarter time and modified lane agility are key coefficients that affect the model which supports with changing a draft position of a player, while max vertical leap and hand width are shown to have a negative impact on draft position. While the model does tell this for linear regression it does not support it being a good model due to the low R-squared score and high mean squared error.*")
st.write("*Insight of random forest classifier: With the random forest classifier, I wanted to see what were some of the important features or variables that affect the draft position of a prospect. From creating the classifier, the R-squared for the model improved a lot more compared to the linear regression, however the r-squared is still shown to be mid as it has a score of 0.58. However for the feature importance that was ran through the classifier showed that weight, wingspan, agility, are shown to be the highest compared to the other variables featured. This shows that NBA teams are looking at NBA prospects that display speed and weight, as they are important towards getting up and down the court and using the weight to help with dominatiing in times of position mismatch. With wingspan, it supports that having a higher winspan is important towards being higher as a player with a higher winspan can perform better on defense when it comes to getting blocks and steals.*")
st.write("*Overall: While the random forest classifier performed much better compared to the linear regression, there needs to be further investigation on how this model can be improved as the r-squared and mean squared error can all be imporved in order to supports these finding to how it can influence draft position.*" )

### End Dashboard ###
st.divider()