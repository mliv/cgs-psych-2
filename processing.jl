
using DataFrames
using CSV

# Appending classification to the CSV data then merging into one file
# -----------------------------------------------------------------------------

# 1 (Sleep stage = W = wake)
dfW = CSV.File("wav/W/audio_recoded.csv") |> DataFrame
dfW[!,"Class"]  = vec(fill(1,(1,size(dfW)[1])))
#show(dfW)

# 2 (Sleep stage = 1)
df1 = CSV.File("wav/1/audio_recoded.csv") |> DataFrame
df1[!,"Class"]  = vec(fill(2,(1,size(df1)[1])))
#show(df1)

# 3 (Sleep stage = 2)
df2 = CSV.File("wav/2/audio_recoded.csv") |> DataFrame
df2[!,"Class"]  = vec(fill(3,(1,size(df2)[1])))
#show(df2)

# 4 (Sleep stage = 3 and 4)
df34 = CSV.File("wav/34/audio_recoded.csv") |> DataFrame 
df34[!,"Class"] = vec(fill(4,(1,size(df34)[1])))
#show(df34)

# 5 (Sleep stage = R = REM)
dfR = CSV.File("wav/R/audio_recoded.csv") |> DataFrame
dfR[!,"Class"]  = vec(fill(5,(1,size(dfR)[1])))
#show(dfR)

dfCombined = vcat(dfW, df1, df2, df34, dfR)
#show(dfCombined)
CSV.write("audio_recoded_combined_classed.csv", dfCombined)


# Alternative with binary classification (0 = not REM, 1 = REM)
# -----------------------------------------------------------------------------
# 1 (Sleep stage = W = wake)
dfW = CSV.File("wav/W/audio_recoded.csv") |> DataFrame
dfW[!,"Class"]  = vec(fill(0,(1,size(dfW)[1])))
# 2 (Sleep stage = 1)
df1 = CSV.File("wav/1/audio_recoded.csv") |> DataFrame
df1[!,"Class"]  = vec(fill(0,(1,size(df1)[1])))
# 3 (Sleep stage = 2)
df2 = CSV.File("wav/2/audio_recoded.csv") |> DataFrame
df2[!,"Class"]  = vec(fill(0,(1,size(df2)[1])))
# 4 (Sleep stage = 3 and 4)
df34 = CSV.File("wav/34/audio_recoded.csv") |> DataFrame 
df34[!,"Class"] = vec(fill(0,(1,size(df34)[1])))
# 5 (Sleep stage = R = REM)
dfR = CSV.File("wav/R/audio_recoded.csv") |> DataFrame
dfR[!,"Class"]  = vec(fill(1,(1,size(dfR)[1])))
# Combining
dfCombinedBinary = vcat(dfW, df1, df2, df34, dfR)
#show(dfCombined)
CSV.write("audio_recoded_combined_binary.csv", dfCombinedBinary)


# Alternative as 2 files (positives and negatives)
# -----------------------------------------------------------------------------
# 1 (Sleep stage = W = wake)
dfW = CSV.File("wav/W/audio_recoded.csv") |> DataFrame
# 2 (Sleep stage = 1)
df1 = CSV.File("wav/1/audio_recoded.csv") |> DataFrame
# 3 (Sleep stage = 2)
df2 = CSV.File("wav/2/audio_recoded.csv") |> DataFrame
# 4 (Sleep stage = 3 and 4)
df34 = CSV.File("wav/34/audio_recoded.csv") |> DataFrame 
# 5 (Sleep stage = R = REM)
dfR = CSV.File("wav/R/audio_recoded.csv") |> DataFrame
# Combining
dfNegative = vcat(dfW, df1, df2, df34)
dfPositive = dfR
dfBareAll  = vcat(dfW, df1, df2, df34, dfR)
CSV.write("audio_recoded_combined_NonREM.csv", dfNegative)
CSV.write("audio_recoded_combined_REM.csv",    dfPositive)
CSV.write("audio_recoded_combined_bare.csv",   dfBareAll)