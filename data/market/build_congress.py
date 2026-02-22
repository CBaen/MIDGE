#!/usr/bin/env python3
"""
Temporary build script — generates congress_members.json from hardcoded API data.
Run once: python data/market/build_congress.py
Delete after use.
"""

import json
import re
from pathlib import Path

OUTPUT = Path(__file__).parent / "congress_members.json"

# ─── SENATORS WITH FULL COMMITTEE DATA (119th Congress, senate.gov) ──────────

SENATE_DATA = [
    {"last": "Alsobrooks", "first": "Angela D.", "party": "D", "state": "MD", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Environment and Public Works", "Senate Health, Education, Labor, and Pensions", "Senate Special Committee on Aging"]},
    {"last": "Baldwin", "first": "Tammy", "party": "D", "state": "WI", "committees": ["Senate Appropriations", "Senate Commerce, Science, and Transportation", "Senate Health, Education, Labor, and Pensions"]},
    {"last": "Banks", "first": "Jim", "party": "R", "state": "IN", "committees": ["Senate Armed Services", "Senate Banking, Housing, and Urban Affairs", "Senate Health, Education, Labor, and Pensions", "Senate Veterans' Affairs"]},
    {"last": "Barrasso", "first": "John", "party": "R", "state": "WY", "committees": ["Senate Energy and Natural Resources", "Senate Finance", "Senate Foreign Relations"]},
    {"last": "Bennet", "first": "Michael F.", "party": "D", "state": "CO", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Finance", "Senate Rules and Administration", "Senate Select Committee on Intelligence"]},
    {"last": "Blackburn", "first": "Marsha", "party": "R", "state": "TN", "committees": ["Senate Commerce, Science, and Transportation", "Senate Finance", "Senate Judiciary", "Senate Veterans' Affairs", "Joint Economic Committee"]},
    {"last": "Blumenthal", "first": "Richard", "party": "D", "state": "CT", "committees": ["Senate Armed Services", "Senate Homeland Security and Governmental Affairs", "Senate Judiciary", "Senate Veterans' Affairs"]},
    {"last": "Blunt Rochester", "first": "Lisa", "party": "D", "state": "DE", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Commerce, Science, and Transportation", "Senate Environment and Public Works", "Senate Health, Education, Labor, and Pensions"]},
    {"last": "Booker", "first": "Cory A.", "party": "D", "state": "NJ", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Foreign Relations", "Senate Small Business and Entrepreneurship", "Senate Judiciary"]},
    {"last": "Boozman", "first": "John", "party": "R", "state": "AR", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Appropriations", "Senate Environment and Public Works", "Senate Rules and Administration", "Senate Veterans' Affairs"]},
    {"last": "Britt", "first": "Katie Boyd", "party": "R", "state": "AL", "committees": ["Senate Appropriations", "Senate Banking, Housing, and Urban Affairs", "Senate Rules and Administration", "Senate Judiciary"]},
    {"last": "Budd", "first": "Ted", "party": "R", "state": "NC", "committees": ["Senate Armed Services", "Senate Commerce, Science, and Transportation", "Senate Small Business and Entrepreneurship", "Senate Select Committee on Intelligence"]},
    {"last": "Cantwell", "first": "Maria", "party": "D", "state": "WA", "committees": ["Senate Commerce, Science, and Transportation", "Senate Energy and Natural Resources", "Senate Finance", "Senate Indian Affairs", "Senate Small Business and Entrepreneurship"]},
    {"last": "Capito", "first": "Shelley Moore", "party": "R", "state": "WV", "committees": ["Senate Appropriations", "Senate Commerce, Science, and Transportation", "Senate Environment and Public Works", "Senate Rules and Administration"]},
    {"last": "Cassidy", "first": "Bill", "party": "R", "state": "LA", "committees": ["Senate Energy and Natural Resources", "Senate Finance", "Senate Health, Education, Labor, and Pensions", "Senate Veterans' Affairs"]},
    {"last": "Collins", "first": "Susan M.", "party": "R", "state": "ME", "committees": ["Senate Appropriations", "Senate Health, Education, Labor, and Pensions", "Senate Select Committee on Intelligence"]},
    {"last": "Coons", "first": "Christopher A.", "party": "D", "state": "DE", "committees": ["Senate Appropriations", "Senate Foreign Relations", "Senate Small Business and Entrepreneurship", "Senate Judiciary", "Senate Select Committee on Ethics"]},
    {"last": "Cornyn", "first": "John", "party": "R", "state": "TX", "committees": ["Senate Finance", "Senate Foreign Relations", "Senate Budget", "Senate Judiciary", "Senate Select Committee on Intelligence"]},
    {"last": "Cortez Masto", "first": "Catherine", "party": "D", "state": "NV", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Energy and Natural Resources", "Senate Finance", "Senate Indian Affairs"]},
    {"last": "Cotton", "first": "Tom", "party": "R", "state": "AR", "committees": ["Senate Armed Services", "Senate Energy and Natural Resources", "Senate Select Committee on Intelligence"]},
    {"last": "Cramer", "first": "Kevin", "party": "R", "state": "ND", "committees": ["Senate Armed Services", "Senate Banking, Housing, and Urban Affairs", "Senate Environment and Public Works", "Senate Veterans' Affairs"]},
    {"last": "Crapo", "first": "Mike", "party": "R", "state": "ID", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Finance", "Senate Budget"]},
    {"last": "Cruz", "first": "Ted", "party": "R", "state": "TX", "committees": ["Senate Commerce, Science, and Transportation", "Senate Foreign Relations", "Senate Rules and Administration", "Senate Judiciary"]},
    {"last": "Curtis", "first": "John R.", "party": "R", "state": "UT", "committees": ["Senate Commerce, Science, and Transportation", "Senate Environment and Public Works", "Senate Foreign Relations", "Senate Small Business and Entrepreneurship"]},
    {"last": "Daines", "first": "Steve", "party": "R", "state": "MT", "committees": ["Senate Energy and Natural Resources", "Senate Finance", "Senate Foreign Relations", "Senate Indian Affairs"]},
    {"last": "Duckworth", "first": "Tammy", "party": "D", "state": "IL", "committees": ["Senate Armed Services", "Senate Commerce, Science, and Transportation", "Senate Foreign Relations", "Senate Veterans' Affairs"]},
    {"last": "Durbin", "first": "Richard J.", "party": "D", "state": "IL", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Appropriations", "Senate Judiciary"]},
    {"last": "Ernst", "first": "Joni", "party": "R", "state": "IA", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Armed Services", "Senate Homeland Security and Governmental Affairs", "Senate Small Business and Entrepreneurship"]},
    {"last": "Fetterman", "first": "John", "party": "D", "state": "PA", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Commerce, Science, and Transportation", "Senate Homeland Security and Governmental Affairs"]},
    {"last": "Fischer", "first": "Deb", "party": "R", "state": "NE", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Appropriations", "Senate Armed Services", "Senate Commerce, Science, and Transportation", "Senate Rules and Administration", "Senate Select Committee on Ethics"]},
    {"last": "Gallego", "first": "Ruben", "party": "D", "state": "AZ", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Energy and Natural Resources", "Senate Homeland Security and Governmental Affairs", "Senate Veterans' Affairs"]},
    {"last": "Gillibrand", "first": "Kirsten E.", "party": "D", "state": "NY", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Select Committee on Intelligence", "Senate Special Committee on Aging"]},
    {"last": "Graham", "first": "Lindsey", "party": "R", "state": "SC", "committees": ["Senate Appropriations", "Senate Environment and Public Works", "Senate Budget", "Senate Judiciary"]},
    {"last": "Grassley", "first": "Chuck", "party": "R", "state": "IA", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Finance", "Senate Budget", "Senate Judiciary"]},
    {"last": "Hagerty", "first": "Bill", "party": "R", "state": "TN", "committees": ["Senate Appropriations", "Senate Banking, Housing, and Urban Affairs", "Senate Foreign Relations", "Senate Rules and Administration"]},
    {"last": "Hassan", "first": "Margaret Wood", "party": "D", "state": "NH", "committees": ["Senate Finance", "Senate Health, Education, Labor, and Pensions", "Senate Homeland Security and Governmental Affairs", "Senate Veterans' Affairs"]},
    {"last": "Hawley", "first": "Josh", "party": "R", "state": "MO", "committees": ["Senate Health, Education, Labor, and Pensions", "Senate Homeland Security and Governmental Affairs", "Senate Small Business and Entrepreneurship", "Senate Judiciary"]},
    {"last": "Heinrich", "first": "Martin", "party": "D", "state": "NM", "committees": ["Senate Appropriations", "Senate Energy and Natural Resources", "Senate Select Committee on Intelligence"]},
    {"last": "Hickenlooper", "first": "John W.", "party": "D", "state": "CO", "committees": ["Senate Commerce, Science, and Transportation", "Senate Energy and Natural Resources", "Senate Health, Education, Labor, and Pensions", "Senate Small Business and Entrepreneurship"]},
    {"last": "Hirono", "first": "Mazie K.", "party": "D", "state": "HI", "committees": ["Senate Armed Services", "Senate Energy and Natural Resources", "Senate Small Business and Entrepreneurship", "Senate Judiciary", "Senate Veterans' Affairs"]},
    {"last": "Hoeven", "first": "John", "party": "R", "state": "ND", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Appropriations", "Senate Energy and Natural Resources", "Senate Indian Affairs"]},
    {"last": "Husted", "first": "Jon", "party": "R", "state": "OH", "committees": ["Senate Environment and Public Works", "Senate Health, Education, Labor, and Pensions", "Senate Small Business and Entrepreneurship", "Senate Special Committee on Aging"]},
    {"last": "Hyde-Smith", "first": "Cindy", "party": "R", "state": "MS", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Appropriations", "Senate Energy and Natural Resources", "Senate Rules and Administration"]},
    {"last": "Johnson", "first": "Ron", "party": "R", "state": "WI", "committees": ["Senate Finance", "Senate Homeland Security and Governmental Affairs", "Senate Budget", "Senate Special Committee on Aging"]},
    {"last": "Justice", "first": "James C.", "party": "R", "state": "WV", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Energy and Natural Resources", "Senate Small Business and Entrepreneurship", "Senate Special Committee on Aging"]},
    {"last": "Kaine", "first": "Tim", "party": "D", "state": "VA", "committees": ["Senate Armed Services", "Senate Foreign Relations", "Senate Health, Education, Labor, and Pensions", "Senate Budget"]},
    {"last": "Kelly", "first": "Mark", "party": "D", "state": "AZ", "committees": ["Senate Armed Services", "Senate Environment and Public Works", "Senate Select Committee on Intelligence", "Senate Special Committee on Aging"]},
    {"last": "Kennedy", "first": "John", "party": "R", "state": "LA", "committees": ["Senate Appropriations", "Senate Banking, Housing, and Urban Affairs", "Senate Budget", "Senate Judiciary"]},
    {"last": "Kim", "first": "Andy", "party": "D", "state": "NJ", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Commerce, Science, and Transportation", "Senate Health, Education, Labor, and Pensions", "Senate Homeland Security and Governmental Affairs", "Senate Special Committee on Aging"]},
    {"last": "King", "first": "Angus S., Jr.", "party": "I", "state": "ME", "committees": ["Senate Armed Services", "Senate Energy and Natural Resources", "Senate Veterans' Affairs", "Senate Select Committee on Intelligence"]},
    {"last": "Klobuchar", "first": "Amy", "party": "D", "state": "MN", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Commerce, Science, and Transportation", "Senate Rules and Administration", "Senate Judiciary"]},
    {"last": "Lankford", "first": "James", "party": "R", "state": "OK", "committees": ["Senate Finance", "Senate Homeland Security and Governmental Affairs", "Senate Select Committee on Ethics", "Senate Select Committee on Intelligence"]},
    {"last": "Lee", "first": "Mike", "party": "R", "state": "UT", "committees": ["Senate Energy and Natural Resources", "Senate Foreign Relations", "Senate Budget", "Senate Judiciary"]},
    {"last": "Lujan", "first": "Ben Ray", "party": "D", "state": "NM", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Commerce, Science, and Transportation", "Senate Finance", "Senate Indian Affairs", "Senate Budget"]},
    {"last": "Lummis", "first": "Cynthia M.", "party": "R", "state": "WY", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Commerce, Science, and Transportation", "Senate Environment and Public Works"]},
    {"last": "Markey", "first": "Edward J.", "party": "D", "state": "MA", "committees": ["Senate Commerce, Science, and Transportation", "Senate Environment and Public Works", "Senate Health, Education, Labor, and Pensions", "Senate Small Business and Entrepreneurship"]},
    {"last": "Marshall", "first": "Roger", "party": "R", "state": "KS", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Finance", "Senate Health, Education, Labor, and Pensions", "Senate Budget"]},
    {"last": "McConnell", "first": "Mitch", "party": "R", "state": "KY", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Appropriations", "Senate Rules and Administration"]},
    {"last": "McCormick", "first": "David", "party": "R", "state": "PA", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Energy and Natural Resources", "Senate Foreign Relations", "Senate Special Committee on Aging"]},
    {"last": "Merkley", "first": "Jeff", "party": "D", "state": "OR", "committees": ["Senate Appropriations", "Senate Environment and Public Works", "Senate Foreign Relations", "Senate Rules and Administration", "Senate Budget"]},
    {"last": "Moody", "first": "Ashley", "party": "R", "state": "FL", "committees": ["Senate Health, Education, Labor, and Pensions", "Senate Homeland Security and Governmental Affairs", "Senate Judiciary", "Senate Special Committee on Aging"]},
    {"last": "Moran", "first": "Jerry", "party": "R", "state": "KS", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Appropriations", "Senate Commerce, Science, and Transportation", "Senate Indian Affairs", "Senate Veterans' Affairs", "Senate Select Committee on Intelligence"]},
    {"last": "Moreno", "first": "Bernie", "party": "R", "state": "OH", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Commerce, Science, and Transportation", "Senate Homeland Security and Governmental Affairs", "Senate Budget"]},
    {"last": "Mullin", "first": "Markwayne", "party": "R", "state": "OK", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Health, Education, Labor, and Pensions", "Senate Indian Affairs"]},
    {"last": "Murkowski", "first": "Lisa", "party": "R", "state": "AK", "committees": ["Senate Appropriations", "Senate Energy and Natural Resources", "Senate Health, Education, Labor, and Pensions", "Senate Indian Affairs"]},
    {"last": "Murphy", "first": "Christopher", "party": "D", "state": "CT", "committees": ["Senate Appropriations", "Senate Foreign Relations", "Senate Health, Education, Labor, and Pensions"]},
    {"last": "Murray", "first": "Patty", "party": "D", "state": "WA", "committees": ["Senate Appropriations", "Senate Health, Education, Labor, and Pensions", "Senate Budget", "Senate Veterans' Affairs"]},
    {"last": "Ossoff", "first": "Jon", "party": "D", "state": "GA", "committees": ["Senate Appropriations", "Senate Rules and Administration", "Senate Select Committee on Intelligence"]},
    {"last": "Padilla", "first": "Alex", "party": "D", "state": "CA", "committees": ["Senate Energy and Natural Resources", "Senate Environment and Public Works", "Senate Rules and Administration", "Senate Budget", "Senate Judiciary"]},
    {"last": "Paul", "first": "Rand", "party": "R", "state": "KY", "committees": ["Senate Foreign Relations", "Senate Health, Education, Labor, and Pensions", "Senate Homeland Security and Governmental Affairs", "Senate Small Business and Entrepreneurship"]},
    {"last": "Peters", "first": "Gary C.", "party": "D", "state": "MI", "committees": ["Senate Appropriations", "Senate Commerce, Science, and Transportation", "Senate Homeland Security and Governmental Affairs"]},
    {"last": "Reed", "first": "Jack", "party": "D", "state": "RI", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Banking, Housing, and Urban Affairs"]},
    {"last": "Ricketts", "first": "Pete", "party": "R", "state": "NE", "committees": ["Senate Appropriations", "Senate Commerce, Science, and Transportation", "Senate Energy and Natural Resources", "Senate Indian Affairs"]},
    {"last": "Risch", "first": "James E.", "party": "R", "state": "ID", "committees": ["Senate Appropriations", "Senate Foreign Relations", "Senate Select Committee on Intelligence"]},
    {"last": "Rosen", "first": "Jacky", "party": "D", "state": "NV", "committees": ["Senate Appropriations", "Senate Commerce, Science, and Transportation", "Senate Health, Education, Labor, and Pensions"]},
    {"last": "Rounds", "first": "Mike", "party": "R", "state": "SD", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Banking, Housing, and Urban Affairs", "Senate Indian Affairs"]},
    {"last": "Sanders", "first": "Bernard", "party": "I", "state": "VT", "committees": ["Senate Appropriations", "Senate Health, Education, Labor, and Pensions", "Senate Budget"]},
    {"last": "Schatz", "first": "Brian", "party": "D", "state": "HI", "committees": ["Senate Appropriations", "Senate Commerce, Science, and Transportation", "Senate Indian Affairs", "Senate Select Committee on Intelligence"]},
    {"last": "Schiff", "first": "Adam B.", "party": "D", "state": "CA", "committees": ["Senate Appropriations", "Senate Rules and Administration", "Senate Select Committee on Intelligence"]},
    {"last": "Schmitt", "first": "Eric", "party": "R", "state": "MO", "committees": ["Senate Commerce, Science, and Transportation", "Senate Energy and Natural Resources", "Senate Small Business and Entrepreneurship"]},
    {"last": "Schumer", "first": "Charles E.", "party": "D", "state": "NY", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Appropriations", "Senate Rules and Administration"]},
    {"last": "Scott", "first": "Rick", "party": "R", "state": "FL", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Commerce, Science, and Transportation", "Senate Budget"]},
    {"last": "Scott", "first": "Tim", "party": "R", "state": "SC", "committees": ["Senate Finance", "Senate Health, Education, Labor, and Pensions", "Senate Small Business and Entrepreneurship"]},
    {"last": "Shaheen", "first": "Jeanne", "party": "D", "state": "NH", "committees": ["Senate Appropriations", "Senate Energy and Natural Resources", "Senate Foreign Relations", "Senate Select Committee on Intelligence"]},
    {"last": "Sheehy", "first": "Tim", "party": "R", "state": "MT", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Commerce, Science, and Transportation", "Senate Indian Affairs"]},
    {"last": "Slotkin", "first": "Elissa", "party": "D", "state": "MI", "committees": ["Senate Armed Services", "Senate Homeland Security and Governmental Affairs", "Senate Select Committee on Intelligence"]},
    {"last": "Smith", "first": "Tina", "party": "D", "state": "MN", "committees": ["Senate Agriculture, Nutrition, and Forestry", "Senate Health, Education, Labor, and Pensions", "Senate Indian Affairs", "Senate Special Committee on Aging"]},
    {"last": "Sullivan", "first": "Dan", "party": "R", "state": "AK", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Commerce, Science, and Transportation", "Senate Select Committee on Intelligence"]},
    {"last": "Thune", "first": "John", "party": "R", "state": "SD", "committees": ["Senate Appropriations", "Senate Commerce, Science, and Transportation", "Senate Finance"]},
    {"last": "Tillis", "first": "Thom", "party": "R", "state": "NC", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Banking, Housing, and Urban Affairs", "Senate Judiciary"]},
    {"last": "Tuberville", "first": "Tommy", "party": "R", "state": "AL", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Veterans' Affairs"]},
    {"last": "Van Hollen", "first": "Chris", "party": "D", "state": "MD", "committees": ["Senate Appropriations", "Senate Budget"]},
    {"last": "Warner", "first": "Mark R.", "party": "D", "state": "VA", "committees": ["Senate Appropriations", "Senate Finance", "Senate Rules and Administration", "Senate Select Committee on Intelligence"]},
    {"last": "Warnock", "first": "Raphael G.", "party": "D", "state": "GA", "committees": ["Senate Appropriations", "Senate Health, Education, Labor, and Pensions", "Senate Veterans' Affairs"]},
    {"last": "Warren", "first": "Elizabeth", "party": "D", "state": "MA", "committees": ["Senate Banking, Housing, and Urban Affairs", "Senate Health, Education, Labor, and Pensions", "Senate Budget"]},
    {"last": "Welch", "first": "Peter", "party": "D", "state": "VT", "committees": ["Senate Appropriations", "Senate Energy and Natural Resources", "Senate Judiciary"]},
    {"last": "Whitehouse", "first": "Sheldon", "party": "D", "state": "RI", "committees": ["Senate Appropriations", "Senate Environment and Public Works", "Senate Judiciary", "Senate Select Committee on Intelligence"]},
    {"last": "Wicker", "first": "Roger F.", "party": "R", "state": "MS", "committees": ["Senate Appropriations", "Senate Armed Services", "Senate Commerce, Science, and Transportation"]},
    {"last": "Wyden", "first": "Ron", "party": "D", "state": "OR", "committees": ["Senate Appropriations", "Senate Energy and Natural Resources", "Senate Finance"]},
    {"last": "Young", "first": "Todd", "party": "R", "state": "IN", "committees": ["Senate Appropriations", "Senate Foreign Relations", "Senate Budget"]},
]

# ─── REPRESENTATIVES (name/party/state from govtrack — no committee data yet) ──

REP_RAW = [
    # Batch 1 (offset 0)
    ("Robert Aderholt", "R", "AL"), ("Gus Bilirakis", "R", "FL"), ("Sanford Bishop", "D", "GA"),
    ("Vern Buchanan", "R", "FL"), ("Ken Calvert", "R", "CA"), ("Andre Carson", "D", "IN"),
    ("John R. Carter", "R", "TX"), ("Kathy Castor", "D", "FL"), ("Judy Chu", "D", "CA"),
    ("Yvette Clarke", "D", "NY"), ("Emanuel Cleaver", "D", "MO"), ("James Clyburn", "D", "SC"),
    ("Steve Cohen", "D", "TN"), ("Tom Cole", "R", "OK"), ("Jim Costa", "D", "CA"),
    ("Joe Courtney", "D", "CT"), ("Eric Crawford", "R", "AR"), ("Henry Cuellar", "D", "TX"),
    ("Danny Davis", "D", "IL"), ("Diana DeGette", "D", "CO"), ("Rosa DeLauro", "D", "CT"),
    ("Scott DesJarlais", "R", "TN"), ("Mario Diaz-Balart", "R", "FL"), ("Lloyd Doggett", "D", "TX"),
    ("Chuck Fleischmann", "R", "TN"), ("Virginia Foxx", "R", "NC"), ("John Garamendi", "D", "CA"),
    ("Paul Gosar", "R", "AZ"), ("Sam Graves", "R", "MO"), ("Al Green", "D", "TX"),
    ("Morgan Griffith", "R", "VA"), ("Brett Guthrie", "R", "KY"), ("Andy Harris", "R", "MD"),
    ("James Himes", "D", "CT"), ("Steny Hoyer", "D", "MD"), ("Bill Huizenga", "R", "MI"),
    ("Hank Johnson", "D", "GA"), ("Jim Jordan", "R", "OH"), ("Marcy Kaptur", "D", "OH"),
    ("William Keating", "D", "MA"), ("Mike Kelly", "R", "PA"), ("Rick Larsen", "D", "WA"),
    ("John Larson", "D", "CT"), ("Robert Latta", "R", "OH"), ("Zoe Lofgren", "D", "CA"),
    ("Frank Lucas", "R", "OK"), ("Stephen Lynch", "D", "MA"), ("Doris Matsui", "D", "CA"),
    ("Michael McCaul", "R", "TX"), ("Tom McClintock", "R", "CA"), ("Betty McCollum", "D", "MN"),
    ("James McGovern", "D", "MA"), ("Gregory Meeks", "D", "NY"), ("Gwen Moore", "D", "WI"),
    ("Jerrold Nadler", "D", "NY"), ("Richard Neal", "D", "MA"), ("Eleanor Holmes Norton", "D", "DC"),
    ("Frank Pallone", "D", "NJ"), ("Nancy Pelosi", "D", "CA"), ("Chellie Pingree", "D", "ME"),
    ("Mike Quigley", "D", "IL"), ("Harold Rogers", "R", "KY"), ("Mike Rogers", "R", "AL"),
    ("Steve Scalise", "R", "LA"), ("Jan Schakowsky", "D", "IL"), ("David Schweikert", "R", "AZ"),
    ("Austin Scott", "R", "GA"), ("David Scott", "D", "GA"), ("Bobby Scott", "D", "VA"),
    ("Terri Sewell", "D", "AL"), ("Brad Sherman", "D", "CA"), ("Mike Simpson", "R", "ID"),
    # Batch 2 (offset 100)
    ("Raul Ruiz", "D", "CA"), ("Mark Takano", "D", "CA"), ("Juan Vargas", "D", "CA"),
    ("Scott Peters", "D", "CA"), ("Lois Frankel", "D", "FL"), ("Garland Barr", "R", "KY"),
    ("Ann Wagner", "R", "MO"), ("Richard Hudson", "R", "NC"), ("Grace Meng", "D", "NY"),
    ("Hakeem Jeffries", "D", "NY"), ("Joyce Beatty", "D", "OH"), ("David Joyce", "R", "OH"),
    ("Scott Perry", "R", "PA"), ("Randy Weber", "R", "TX"), ("Joaquin Castro", "D", "TX"),
    ("Roger Williams", "R", "TX"), ("Marc Veasey", "D", "TX"), ("Mark Pocan", "D", "WI"),
    ("Robin Kelly", "D", "IL"), ("Jason Smith", "R", "MO"), ("Katherine Clark", "D", "MA"),
    ("Donald Norcross", "D", "NJ"), ("Alma Adams", "D", "NC"), ("Gary Palmer", "R", "AL"),
    ("French Hill", "R", "AR"), ("Bruce Westerman", "R", "AR"), ("Mark DeSaulnier", "D", "CA"),
    ("Pete Aguilar", "D", "CA"), ("Ted Lieu", "D", "CA"), ("Norma Torres", "D", "CA"),
    ("Earl Carter", "R", "GA"), ("Barry Loudermilk", "R", "GA"), ("Rick Allen", "R", "GA"),
    ("Mike Bost", "R", "IL"), ("Seth Moulton", "D", "MA"), ("John Moolenaar", "R", "MI"),
    ("Debbie Dingell", "D", "MI"), ("Tom Emmer", "R", "MN"), ("David Rouzer", "R", "NC"),
    ("Bonnie Watson Coleman", "D", "NJ"), ("Elise Stefanik", "R", "NY"), ("Brendan Boyle", "D", "PA"),
    ("Brian Babin", "R", "TX"), ("Donald Beyer", "D", "VA"), ("Stacey Plaskett", "D", "VI"),
    ("Dan Newhouse", "R", "WA"), ("Glenn Grothman", "R", "WI"), ("Aumua Amata Radewagen", "R", "AS"),
    ("Trent Kelly", "R", "MS"), ("Darin LaHood", "R", "IL"), ("Warren Davidson", "R", "OH"),
    ("James Comer", "R", "KY"), ("Dwight Evans", "D", "PA"), ("Bradley Schneider", "D", "IL"),
    ("Andy Biggs", "R", "AZ"), ("Ro Khanna", "D", "CA"), ("Jimmy Panetta", "D", "CA"),
    ("Salud Carbajal", "D", "CA"), ("Nanette Barragan", "D", "CA"), ("Luis Correa", "D", "CA"),
    ("Neal Dunn", "R", "FL"), ("John Rutherford", "R", "FL"), ("Darren Soto", "D", "FL"),
    ("Brian Mast", "R", "FL"), ("Raja Krishnamoorthi", "D", "IL"), ("Clay Higgins", "R", "LA"),
    ("Mike Johnson", "R", "LA"), ("Jamie Raskin", "D", "MD"), ("Jack Bergman", "R", "MI"),
    ("Don Bacon", "R", "NE"), ("Josh Gottheimer", "D", "NJ"), ("Adriano Espaillat", "D", "NY"),
    ("Brian Fitzpatrick", "R", "PA"), ("Lloyd Smucker", "R", "PA"), ("David Kustoff", "R", "TN"),
    # Batch 3 (offset 200)
    ("James Baird", "R", "IN"), ("Sharice Davids", "D", "KS"), ("Lori Trahan", "D", "MA"),
    ("Ayanna Pressley", "D", "MA"), ("Haley Stevens", "D", "MI"), ("Rashida Tlaib", "D", "MI"),
    ("Angie Craig", "D", "MN"), ("Ilhan Omar", "D", "MN"), ("Pete Stauber", "R", "MN"),
    ("Michael Guest", "R", "MS"), ("Chris Pappas", "D", "NH"), ("Jefferson Van Drew", "R", "NJ"),
    ("Susie Lee", "D", "NV"), ("Alexandria Ocasio-Cortez", "D", "NY"), ("Madeleine Dean", "D", "PA"),
    ("Chrissy Houlahan", "D", "PA"), ("Daniel Meuser", "R", "PA"), ("John Joyce", "R", "PA"),
    ("Guy Reschenthaler", "R", "PA"), ("William Timmons", "R", "SC"), ("Dusty Johnson", "R", "SD"),
    ("Tim Burchett", "R", "TN"), ("John W. Rose", "R", "TN"), ("Dan Crenshaw", "R", "TX"),
    ("Lance Gooden", "R", "TX"), ("Lizzie Fletcher", "D", "TX"), ("Veronica Escobar", "D", "TX"),
    ("Chip Roy", "R", "TX"), ("Sylvia Garcia", "D", "TX"), ("Ben Cline", "R", "VA"),
    ("Kim Schrier", "D", "WA"), ("Bryan Steil", "R", "WI"), ("Carol Miller", "R", "WV"),
    ("Jared Golden", "D", "ME"), ("Gregory Murphy", "R", "NC"), ("Kweisi Mfume", "D", "MD"),
    ("Thomas Tiffany", "R", "WI"), ("Darrell Issa", "R", "CA"), ("Pete Sessions", "R", "TX"),
    ("David Valadao", "R", "CA"), ("Barry Moore", "R", "AL"), ("Jay Obernolte", "R", "CA"),
    ("Young Kim", "R", "CA"), ("Sara Jacobs", "D", "CA"), ("Lauren Boebert", "R", "CO"),
    ("Katherine Cammack", "R", "FL"), ("Scott Franklin", "R", "FL"), ("Byron Donalds", "R", "FL"),
    ("Carlos Gimenez", "R", "FL"), ("Maria Salazar", "R", "FL"), ("Nikema Williams", "D", "GA"),
    ("Andrew Clyde", "R", "GA"), ("Ashley Hinson", "R", "IA"), ("Mariannette Miller-Meeks", "R", "IA"),
    ("Randy Feenstra", "R", "IA"), ("Mary Miller", "R", "IL"), ("Frank Mrvan", "D", "IN"),
    ("Victoria Spartz", "R", "IN"), ("Tracey Mann", "R", "KS"), ("Jake Auchincloss", "D", "MA"),
    ("Lisa McClain", "R", "MI"), ("Michelle Fischbach", "R", "MN"), ("Deborah Ross", "D", "NC"),
    ("Teresa Leger Fernandez", "D", "NM"), ("Andrew Garbarino", "R", "NY"),
    ("Nicole Malliotakis", "R", "NY"), ("Ritchie Torres", "D", "NY"), ("Stephanie Bice", "R", "OK"),
    ("Cliff Bentz", "R", "OR"), ("Nancy Mace", "R", "SC"), ("Diana Harshbarger", "R", "TN"),
    ("Patrick Fallon", "R", "TX"), ("August Pfluger", "R", "TX"), ("Ronny Jackson", "R", "TX"),
    ("Troy Nehls", "R", "TX"), ("Tony Gonzales", "R", "TX"),
    # Batch 4 (offset 300)
    ("Robert Garcia", "D", "CA"), ("Brittany Pettersen", "D", "CO"), ("Aaron Bean", "R", "FL"),
    ("Cory Mills", "R", "FL"), ("Maxwell Frost", "D", "FL"), ("Anna Luna", "R", "FL"),
    ("Laurel Lee", "R", "FL"), ("Jared Moskowitz", "D", "FL"), ("Rich McCormick", "R", "GA"),
    ("Mike Collins", "R", "GA"), ("James Moylan", "R", "GU"), ("Jill Tokuda", "D", "HI"),
    ("Zachary Nunn", "R", "IA"), ("Jonathan Jackson", "D", "IL"), ("Delia Ramirez", "D", "IL"),
    ("Nicole Budzinski", "D", "IL"), ("Eric Sorensen", "D", "IL"), ("Erin Houchin", "R", "IN"),
    ("Morgan McGarvey", "D", "KY"), ("Glenn Ivey", "D", "MD"), ("Hillary Scholten", "D", "MI"),
    ("John James", "R", "MI"), ("Shri Thanedar", "D", "MI"), ("Mark Alford", "R", "MO"),
    ("Eric Burlison", "R", "MO"), ("Mike Ezell", "R", "MS"), ("Donald Davis", "D", "NC"),
    ("Valerie Foushee", "D", "NC"), ("Chuck Edwards", "R", "NC"), ("Thomas Kean", "R", "NJ"),
    ("Robert Menendez", "D", "NJ"), ("Gabriel Vasquez", "D", "NM"), ("Nicolas LaLota", "R", "NY"),
    ("Dan Goldman", "D", "NY"), ("Michael Lawler", "R", "NY"), ("Nicholas Langworthy", "R", "NY"),
    ("Greg Landsman", "D", "OH"), ("Max Miller", "R", "OH"), ("Emilia Sykes", "D", "OH"),
    ("Josh Brecheen", "R", "OK"), ("Valerie Hoyle", "D", "OR"), ("Andrea Salinas", "D", "OR"),
    ("Summer Lee", "D", "PA"), ("Chris Deluzio", "D", "PA"), ("Seth Magaziner", "D", "RI"),
    ("Russell Fry", "R", "SC"), ("Andrew Ogles", "R", "TN"), ("Nathaniel Moran", "R", "TX"),
    ("Keith Self", "R", "TX"), ("Morgan Luttrell", "R", "TX"), ("Monica De La Cruz", "R", "TX"),
    ("Jasmine Crockett", "D", "TX"), ("Gregorio Casar", "D", "TX"), ("Wesley Hunt", "R", "TX"),
    ("Jennifer Kiggans", "R", "VA"), ("Becca Balint", "D", "VT"),
    ("Marie Gluesenkamp Perez", "D", "WA"), ("Derrick Van Orden", "R", "WI"),
    ("Harriet Hageman", "R", "WY"), ("Jennifer McClellan", "D", "VA"), ("Gabe Amo", "D", "RI"),
    ("Celeste Maloy", "R", "UT"), ("Thomas Suozzi", "D", "NY"), ("Timothy Kennedy", "D", "NY"),
    ("Vince Fong", "R", "CA"), ("Michael Rulli", "R", "OH"), ("LaMonica McIver", "D", "NJ"),
    ("Tony Wied", "R", "WI"), ("Cleo Fields", "D", "LA"), ("Marlin Stutzman", "R", "IN"),
    ("Gilbert Cisneros", "D", "CA"), ("Nicholas Begich III", "R", "AK"),
    ("Shomari Figures", "D", "AL"), ("Yassamin Ansari", "D", "AZ"), ("Abraham Hamadeh", "R", "AZ"),
    ("Lateefah Simon", "D", "CA"), ("Adam Gray", "D", "CA"),
    # Batch 5 (offset 400) - newest members
    ("Kimberlyn King-Hinds", "R", "MP"), ("Troy Downing", "R", "MT"),
    ("Addison McDowell", "R", "NC"), ("Mark Everette Harris", "R", "NC"),
    ("Pat Harrigan", "R", "NC"), ("Brad Knott", "R", "NC"), ("Tim Moore", "R", "NC"),
    ("Julie Fedorchak", "R", "ND"), ("Maggie Goodlander", "D", "NH"),
    ("Herbert Conaway", "D", "NJ"), ("Nellie Pou", "D", "NJ"), ("Laura Gillen", "D", "NY"),
    ("George Latimer", "D", "NY"), ("Josh Riley", "D", "NY"), ("John Mannion", "D", "NY"),
    ("David Taylor", "R", "OH"), ("Maxine Dexter", "D", "OR"), ("Janelle Bynum", "D", "OR"),
    ("Ryan Mackenzie", "R", "PA"), ("Robert Bresnahan", "R", "PA"),
    ("Pablo Jose Hernandez Rivera", "D", "PR"), ("Sheri Biggs", "R", "SC"),
    ("Craig Goldman", "R", "TX"), ("Brandon Gill", "R", "TX"), ("Julie Johnson", "D", "TX"),
    ("Mike Kennedy", "R", "UT"), ("John J. McGuire", "R", "VA"), ("Eugene Vindman", "D", "VA"),
    ("Suhas Subramanyam", "D", "VA"), ("Michael Baumgartner", "R", "WA"),
    ("Emily Randall", "D", "WA"), ("Riley Moore", "R", "WV"), ("Jimmy Patronis", "R", "FL"),
    ("Randy Fine", "R", "FL"), ("James Walkinshaw", "D", "VA"), ("Adelita Grijalva", "D", "AZ"),
    ("Matt Van Epps", "R", "TN"), ("Christian Menefee", "D", "TX"),
]


def extract_last_name(full_name):
    """Get the last name token for use as dictionary key."""
    # Handle compound names like "Ocasio-Cortez", "Van Drew", "Leger Fernandez"
    parts = full_name.strip().split()
    if not parts:
        return "UNKNOWN"
    return parts[-1].upper()


def clean_committee_name(raw):
    """Normalize committee name: strip 'Committee on', prefix Chamber."""
    return raw.strip()


def make_search_keys(first, last, role_type):
    """
    Build a list of tokens that should trigger a match for this person.
    politician_tracker._identify_politician checks: key in filer_upper (substring).
    We want: compound last name with space, compound last name no space,
    individual tokens of compound names, and the primary key form.
    """
    keys = set()
    last_parts = last.split()
    # Full last name with space (e.g. "CORTEZ MASTO") — matches STOCK Act disclosures
    keys.add(last.upper())
    # Full last name with underscore (primary key form)
    keys.add(last.upper().replace(" ", "_"))
    # Full last name no space (e.g. "CORTEZMASTO")
    keys.add(last.upper().replace(" ", ""))
    # Each word of last name individually (e.g. "MASTO", "CORTEZ")
    for part in last_parts:
        keys.add(part.upper())
    return sorted(keys)


def build_members():
    members = {}

    # ── Senators ──────────────────────────────────────────────────────────────
    for s in SENATE_DATA:
        last = s["last"]
        last_upper = last.upper()
        # Primary key: underscored compound last name
        primary_key = last_upper.replace(" ", "_")
        full_name = f"{s['first']} {s['last']}"
        search_keys = make_search_keys(s["first"], last, "senator")
        entry = {
            "name": full_name,
            "party": s["party"],
            "state": s["state"],
            "role_type": "senator",
            "committees": s["committees"],
            "search_keys": search_keys,
        }
        members[primary_key] = entry

        # Add alias keys for compound last names so substring matching works
        # e.g. CORTEZ_MASTO also registered as "CORTEZ MASTO" (space) and last token
        if " " in last:
            # Alias: no-space version
            alias_nospace = last_upper.replace(" ", "")
            if alias_nospace not in members:
                members[alias_nospace] = entry
            # Alias: last word only (e.g. MASTO, ROCHESTER, HOLLEN)
            last_word = last_upper.split()[-1]
            if last_word not in members:
                members[last_word] = entry

    # ── Representatives ───────────────────────────────────────────────────────
    seen_keys = set(members.keys())
    for (full_name, party, state) in REP_RAW:
        parts = full_name.strip().split()
        last_raw = parts[-1] if parts else "UNKNOWN"
        last = last_raw.upper()

        # Disambiguate duplicate last names
        if last in seen_keys:
            key = f"{last}_{state}"
        else:
            key = last

        if key in seen_keys:
            first_initial = parts[0][0].upper() if parts else "X"
            key = f"{last}_{state}_{first_initial}"

        seen_keys.add(key)
        search_keys = make_search_keys(parts[0] if parts else "", last_raw, "representative")
        members[key] = {
            "name": full_name,
            "party": party,
            "state": state,
            "role_type": "representative",
            "committees": [],  # House committee data not yet populated
            "search_keys": search_keys,
        }

    return members


def main():
    members = build_members()

    output = {
        "last_updated": "2026-02-22",
        "source": "govtrack.us API (members) + senate.gov committee assignments (119th Congress)",
        "notes": "Senator committees: complete (119th Congress). Representative committees: empty (pending — house committee data requires separate fetch).",
        "counts": {
            "senators": sum(1 for m in members.values() if m["role_type"] == "senator"),
            "representatives": sum(1 for m in members.values() if m["role_type"] == "representative"),
            "total": len(members),
        },
        "members": members,
    }

    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Written: {OUTPUT}")
    print(f"  Senators:        {output['counts']['senators']}")
    print(f"  Representatives: {output['counts']['representatives']}")
    print(f"  Total:           {output['counts']['total']}")

    # Spot-check key lookup
    for test_key in ["PELOSI", "TUBERVILLE", "OSSOFF", "OCASIO-CORTEZ"]:
        if test_key in members:
            print(f"  [{test_key}] -> {members[test_key]['name']} ({members[test_key]['party']}-{members[test_key]['state']})")
        else:
            # Try alternate forms
            hits = [k for k in members if test_key in k]
            print(f"  [{test_key}] -> not found as exact key; alts: {hits}")


if __name__ == "__main__":
    main()
