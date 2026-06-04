import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px  

st.sidebar.title("📊 Marketing ML App")
st.sidebar.markdown("**Brands:** Nykaa | Purplle | Tira")

page = st.sidebar.radio(
    "Navigate",
    ["Home","EDA","Predict Revenue", "Predict Profit/Loss"]
)

if page == "Home":
    st.title("🛍️ Marketing Campaign Performance Prediction")


elif page == "EDA":
    selection = st.pills(
    "**Insights:**", 
     options=["ROI by Campaign Type and Campaign","profit vs Loss count by Brand","correlation analysis","Regression Model Data","Residuals chart","Features Analysis","Model Comparison"]
)
    df = pd.read_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\cleaned_data\nykaa_purplle_tira_combined_files\nykaa_purplle_tira_eda_df.csv") 
    df1 = pd.read_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\cleaned_data\nykaa_purplle_tira_combined_files\nykaa_purplle_tira_df.csv")
    
    if selection == "ROI by Campaign Type and Campaign":
        
        roi_campaign_type_analysis = (
                                df.groupby(["Campaign_Type", "Campaign"])["New_Calc_ROI"].mean().reset_index()
                                        )

        fig = px.bar(roi_campaign_type_analysis, x="Campaign_Type", y="New_Calc_ROI", color="Campaign", barmode="group",
                        title="Average ROI by Campaign Type & Brand",
                        labels={
                            "Campaign_Type": "Campaign Type",
                            "New_Calc_ROI": "ROI",
                            "Campaign": "Brand"
                                }
                        )
        fig.update_layout(width=1100, height=500, legend_title="Brand", title_x = 0.3)
        st.plotly_chart(fig, use_container_width=True)


    elif selection == "profit vs Loss count by Brand":
        profit_loss_campaign_analysis = (
                 df.groupby(["Campaign", "Profit_Loss"]).size().reset_index(name="Count")
                                )

        profit_loss_campaign_analysis["Label"] = (
                            profit_loss_campaign_analysis["Profit_Loss"].map({
                                                1: "Profit",
                                                0: "Loss"
                                                }))

        fig = px.bar(profit_loss_campaign_analysis, x="Campaign", y="Count", color="Label",
                    barmode="group",
                    title="Profit vs Loss Count by Brand",
                    labels={
                            "Campaign": "Brand",
                            "Count": "Count",
                            "Label": "Status"
                        }
                        )

        fig.update_layout(width=900,height=500,legend_title="Status", title_x = 0.3)
        st.plotly_chart(fig, use_container_width=True)

    elif selection == "correlation analysis":
        correlation_columns = ['Impressions','Clicks','Leads','Conversions','Acquisition_Cost','New_Calc_ROI','Revenue',
                        'Profit_Loss','Engagement_Score','Day','Month','Year','Duration','Campaign_Encoded','Channel_Used_Email',
                        'Channel_Used_Facebook','Channel_Used_Google','Channel_Used_Instagram','Channel_Used_WhatsApp',
                        'Channel_Used_YouTube','Target_Audience_Encoded','Language_Encoded','Customer_Segment_Encoded']

        regression_correlation_matrix = df1[correlation_columns].corr()
        regression_correlation_heatmap_fig = px.imshow(regression_correlation_matrix, 
                                                title='Correlation Analysis for Market campaign for Regression Model',
                                                color_continuous_scale="RDBu_r",
                                                color_continuous_midpoint=0,
                                                labels = dict(color="Correlation",fontsize = 10, fontweight = 'bold'),
                                                text_auto='.2f',
                                                aspect="equal"                              
                                                )
        
        regression_correlation_heatmap_fig.update_layout(title_x = 0.3,
                                                        width=800,
                                                        height=800,
                                                        xaxis=dict(tickangle=-90),
                                                        coloraxis_colorbar=dict(
                                                        len=0.9,                      
                                                        yanchor="middle",             
                                                        y=0.5     )
                                                            )
        st.plotly_chart(regression_correlation_heatmap_fig, use_container_width=True)

        st.markdown("Correlation with Market Campaign: ")
        st.code("""Revenue                     1.000000
Conversions                 0.814716
New_Calc_ROI                0.758087
Leads                       0.748316
Clicks                      0.664261
Engagement_Score            0.522004
Impressions                 0.461396
Profit_Loss                 0.227984
Language_Encoded            0.006649
Channel_Used_Email          0.005416
Channel_Used_Instagram      0.005319
Day                         0.003408
Customer_Segment_Encoded    0.003118
Duration                    0.000696
Channel_Used_Facebook       0.000068
Target_Audience_Encoded     0.000031
Channel_Used_Google        -0.000003
Channel_Used_WhatsApp      -0.000476
Month                      -0.000917
Campaign_Encoded           -0.001438
Channel_Used_YouTube       -0.002750
Year                       -0.002944
Acquisition_Cost           -0.377241""")
            

    elif selection == "Regression Model Data":
        col1, col2 = st.columns(2)

        with col1:
            df3 = pd.read_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\cleaned_data\nykaa_purplle_tira_combined_files\reg_evaluation_test_df.csv")
            fig = px.scatter(df3,
                            x="Actual Revenue",
                            y="Predicted Revenue",
                            title="Random Forest - Testing Data Actual vs Predicted Revenue"
                            )

            fig.update_layout(width = 500,height = 600, title_x = 0.2)

            # Add perfect prediction line (y = x)
            fig.add_shape(type="line",
                            x0=df3["Actual Revenue"].min(),
                            y0=df3["Actual Revenue"].min(),
                            x1=df3["Actual Revenue"].max(),
                            y1=df3["Actual Revenue"].max(),
                            line=dict(dash="dash")
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df4 = pd.read_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\cleaned_data\nykaa_purplle_tira_combined_files\reg_evaluation_train_df.csv")
            fig = px.scatter(df4,
                            x="Actual Revenue",
                            y="Predicted Revenue",
                            title="Random Forest - Training Data Actual vs Predicted Revenue"
                            )

            fig.update_layout(width = 500,height = 600, title_x = 0.1)

            # Add perfect prediction line (y = x)
            fig.add_shape(type="line",
                            x0=df4["Actual Revenue"].min(),
                            y0=df4["Actual Revenue"].min(),
                            x1=df4["Actual Revenue"].max(),
                            y1=df4["Actual Revenue"].max(),
                            line=dict(dash="dash")
            )
            st.plotly_chart(fig, use_container_width=True)


    elif selection == "Residuals chart":
        df5 = pd.read_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\cleaned_data\nykaa_purplle_tira_combined_files\residual_df.csv")
        fig = px.scatter(df5, x="Predicted",y="Residual",
                      title="Residuals vs Predicted Values"
                )
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=500, width = 1000, title_x = 0.5)
        st.plotly_chart(fig, use_container_width=True)
    

    elif selection == "Features Analysis":
        df6 = pd.read_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\cleaned_data\nykaa_purplle_tira_combined_files\feature_importance_df.csv")
        top10 = df6.head(10)

        fig = px.bar(   top10,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Top 10 Feature Importance for Revenue Prediction",
                        text_auto=".3f"
                    )
        fig.update_layout(height=700,title_x = 0.3,
                        yaxis={"categoryorder": "total ascending"}
                        )
        st.plotly_chart(fig, use_container_width=True)
    

    elif selection == "Model Comparison":
        col1, col2 = st.columns(2)
        reg_result_df = pd.read_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\cleaned_data\nykaa_purplle_tira_combined_files\reg_results.csv")
        cls_result_df = pd.read_csv(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\cleaned_data\nykaa_purplle_tira_combined_files\cls_results.csv")
        with col1:
            fig = px.bar(reg_result_df, x="Model", y="R2", text_auto=".4f",title="Regression Model Comparison (R² Score)", color='Model')
            fig.update_layout(height=500, width = 800, title_x=0.3)
            st.plotly_chart(fig, use_container_width=True)
        

        with col2:
            cls_long = cls_result_df.melt(   id_vars="Model",
                                             var_name="Metric",
                                             value_name="Score"
                                            )
            fig = px.bar(cls_long,
                        x="Model",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        text_auto=".3f",
                        title="Classification Model Comparison"
                    )
            fig.update_layout(height=500, title_x = 0.3)
            st.plotly_chart(fig, use_container_width=True)

      
elif page == "Predict Revenue":
    st.set_page_config(
    page_title="Revenue Predictor",
    page_icon="💰",
    layout="wide",
                    )
    
    st.title("💰 Revenue Predictor")
    st.markdown("Marketing Campaign Performance · Random Forest Regression · Nykaa · Purplle · Tira")

    rf_model = joblib.load(
         r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\revenue_random_forest_regression_model.pkl"
    )

    scaler_reg = joblib.load(
         r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\revenue_regression_scalar.pkl"
     )

    CAMPAIGN_MAP        = {"Nykaa": 1, "Purplle": 2, "Tira": 0}
    CAMPAIGN_TYPE_MAP   = {"Email": 0, "Influencer": 1, "Paid Ads": 2, "SEO": 3, "Social Media": 4}
    TARGET_AUDIENCE_MAP = {"College Students": 0, "Premium Shoppers": 1, "Tier 2 City Customers": 2,
                       "Working Women": 3, "Youth": 4}
    LANGUAGE_MAP        = {"Bengali": 0, "English": 1,"Hindi": 2,"Tamil": 3}
    CUSTOMER_SEGMENT_MAP = {"College Students": 0, "Premium Shoppers": 1, "Tier 2 City Customers": 2,
                        "Working Women": 3, "Youth": 4}
    CHANNELS = ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]

    with st.form("prediction_form"):
        col_left, col_right = st.columns(2, gap="large")

        with col_left:
            st.markdown('📢 Campaign Details')
            campaign      = st.selectbox("Campaign Brand",      list(CAMPAIGN_MAP.keys()))
            campaign_type = st.selectbox("Campaign Type",       list(CAMPAIGN_TYPE_MAP.keys()))
            channels      = st.multiselect("Channels Used",     CHANNELS)
            duration      = st.number_input("Campaign Duration (days)", min_value=1, max_value=365, value=1)
            st.markdown('')

            st.markdown('🎯 Audience & Segment')
            target_audience   = st.selectbox("Target Audience",    list(TARGET_AUDIENCE_MAP.keys()))
            language          = st.selectbox("Language",           list(LANGUAGE_MAP.keys()))
            customer_segment  = st.selectbox("Customer Segment",   list(CUSTOMER_SEGMENT_MAP.keys()))
            st.markdown('')

            st.markdown('📅 Campaign Date')
            campaign_date = st.date_input("Campaign Start Date")
            st.markdown('')

        with col_right:
            st.markdown('📊 Performance Metrics')
            impressions = st.number_input("Impressions",      min_value=0.00,    value=0.00,    step=1000.00)
            clicks      = st.number_input("Clicks",           min_value=0.00,    value=0.00,    step=100.00)
            leads       = st.number_input("Leads",            min_value=0.00,    value=0.00,    step=10.00)
            conversions = st.number_input("Conversions",      min_value=0.00,    value=0.00,    step=5.00)
            eng_score   = st.number_input("Engagement Score", min_value=0.00,    value=0.00,    step=0.10)
            st.markdown('')

            st.markdown('💵 Financial Inputs')
            acquisition_cost = st.number_input("Acquisition Cost (₹)", min_value=0.00, 
                                               value=1.00, step=500.00, format="%.2f")
            
            new_calc_roi     = st.number_input("Calculated ROI", value=1.50, step=0.10, format="%.2f",
                                           help="(Revenue - Acquisition_Cost) / Acquisition_Cost")
            st.markdown('')

        submitted = st.form_submit_button("Predict Revenue")

    if submitted:
    # One-hot encode channels
        channel_flags = {f"Channel_Used_{ch}": (1 if ch in channels else 0) for ch in CHANNELS}

        feature_row = {
        "Impressions":              impressions,
        "Clicks":                   clicks,
        "Leads":                    leads,
        "Acquisition_Cost":         acquisition_cost,
        "New_Calc_ROI":             new_calc_roi,
        "Engagement_Score":         eng_score,
        "Conversions":              conversions,
        "Campaign_Encoded":         CAMPAIGN_MAP[campaign],
        "Campaign_Type_Encoded":    CAMPAIGN_TYPE_MAP[campaign_type],
        "Target_Audience_Encoded":  TARGET_AUDIENCE_MAP[target_audience],
        "Language_Encoded":         LANGUAGE_MAP[language],
        "Customer_Segment_Encoded": CUSTOMER_SEGMENT_MAP[customer_segment],
        "Channel_Used_Email":       channel_flags["Channel_Used_Email"],
        "Channel_Used_Facebook":    channel_flags["Channel_Used_Facebook"],
        "Channel_Used_Google":      channel_flags["Channel_Used_Google"],
        "Channel_Used_Instagram":   channel_flags["Channel_Used_Instagram"],
        "Channel_Used_WhatsApp":    channel_flags["Channel_Used_WhatsApp"],
        "Channel_Used_YouTube":     channel_flags["Channel_Used_YouTube"],
        "Duration":                 duration,
        "Day":                      campaign_date.day,
        "Month":                    campaign_date.month,
        "Year":                     campaign_date.year,
             }

        reg_features = [
        'Impressions', 'Clicks', 'Leads', 'Acquisition_Cost', 'New_Calc_ROI',
        'Engagement_Score', 'Conversions', 'Campaign_Encoded', 'Campaign_Type_Encoded',
        'Target_Audience_Encoded', 'Language_Encoded', 'Customer_Segment_Encoded',
        'Channel_Used_Email', 'Channel_Used_Facebook', 'Channel_Used_Google',
        'Channel_Used_Instagram', 'Channel_Used_WhatsApp', 'Channel_Used_YouTube',
        'Duration', 'Day', 'Month', 'Year'
                     ]

        X_input = pd.DataFrame([feature_row])[reg_features]
        X_scaled = scaler_reg.transform(X_input)
        predicted_revenue = rf_model.predict(X_scaled)[0]

        profit          = predicted_revenue - acquisition_cost
        roi_pct         = (profit / acquisition_cost * 100) if acquisition_cost > 0 else 0
        profit_or_loss  = "Profit 📈" if profit > 0 else "Loss 📉"

        st.header("**Predicted Revenue**")
        st.header(f"**₹ {predicted_revenue:,.2f}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Acquisition Cost", f"₹ {acquisition_cost:,.2f}")
        m2.metric("Estimated Profit / Loss", f"₹ {profit:,.2f}")
        m3.metric("ROI", f"{roi_pct:.1f}%")
        m4.metric("Outcome", profit_or_loss)

        with st.expander("📋 Input Summary"):
            summary_df = pd.DataFrame({
                "Feature": list(feature_row.keys()),
                "Value":   list(feature_row.values())
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Random Forest Regression · Marketing Campaign Revenue Prediction · Nykaa · Purplle · Tira")



#=========================
# Predict Profit/Loss:
#=========================

elif page == "Predict Profit/Loss":
    st.set_page_config(
    page_title="Profit / Loss Predictor",
    page_icon="📊",
    layout="wide",
                    )
    
    st.header("📊 Profit / Loss Predictor")
    st.markdown("""Marketing Campaign Performance · Logistic Regression Classification · Nykaa · Purplle · Tira""", )

    log_model  = joblib.load(
        r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\profit_loss_logistic_classification_model.pkl"
        )
    scaler_cls = joblib.load(
        r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\profit_loss_classification_scaler.pkl"
        )

    CAMPAIGN_MAP        = {"Nykaa": 1, "Purplle": 2, "Tira": 0}
    CAMPAIGN_TYPE_MAP   = {"Email": 0, "Influencer": 1, "Paid Ads": 2, "SEO": 3, "Social Media": 4}
    TARGET_AUDIENCE_MAP = {"College Students": 0, "Premium Shoppers": 1, "Tier 2 City Customers": 2,
                            "Working Women": 3, "Youth": 4}
    LANGUAGE_MAP        = {"Bengali": 0, "English": 1,"Hindi": 2,"Tamil": 3}
    CUSTOMER_SEGMENT_MAP = {"College Students": 0, "Premium Shoppers": 1, "Tier 2 City Customers": 2,
                            "Working Women": 3, "Youth": 4}
    CHANNELS = ["Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"]

    CLS_FEATURES = [
    'Impressions', 'Clicks', 'Leads', 'Acquisition_Cost', 'New_Calc_ROI',
    'Engagement_Score', 'Conversions', 'Campaign_Encoded', 'Campaign_Type_Encoded',
    'Target_Audience_Encoded', 'Language_Encoded', 'Customer_Segment_Encoded',
    'Channel_Used_Email', 'Channel_Used_Facebook', 'Channel_Used_Google',
    'Channel_Used_Instagram', 'Channel_Used_WhatsApp', 'Channel_Used_YouTube',
    'Duration', 'Day', 'Month', 'Year'
                    ]


    with st.form("classification_form"):
        col_left, col_right = st.columns(2, gap="large")
        with col_left:
            st.markdown('📢 Campaign Details')
            campaign      = st.selectbox("Campaign Brand",      list(CAMPAIGN_MAP.keys()))
            campaign_type = st.selectbox("Campaign Type",       list(CAMPAIGN_TYPE_MAP.keys()))
            channels      = st.multiselect("Channels Used",     CHANNELS)
            duration      = st.number_input("Campaign Duration (days)", min_value=1, max_value=365, value=30)
            st.markdown('')

            st.markdown('🎯 Audience & Segment')
            target_audience  = st.selectbox("Target Audience",   list(TARGET_AUDIENCE_MAP.keys()))
            language         = st.selectbox("Language",          list(LANGUAGE_MAP.keys()))
            customer_segment = st.selectbox("Customer Segment",  list(CUSTOMER_SEGMENT_MAP.keys()))
            st.markdown('')

            st.markdown('📅 Campaign Date')
            campaign_date = st.date_input("Campaign Start Date")
            st.markdown('')

        with col_right:
            st.markdown('📈 Performance Metrics')
            impressions = st.number_input("Impressions",      min_value=0.00,   value=0.00,  step=1000.0)
            clicks      = st.number_input("Clicks",           min_value=0.00,   value=0.00,  step=100.0)
            leads       = st.number_input("Leads",            min_value=0.00,   value=0.00,  step=10.0)
            conversions = st.number_input("Conversions",      min_value=0.00,   value=0.00,  step=5.0)
            eng_score   = st.number_input("Engagement Score", min_value=0.00,   value=0.00,  step=0.1)
            st.markdown('')

            st.markdown('💵 Financial Inputs')
            acquisition_cost = st.number_input("Acquisition Cost (₹)", min_value=0.01,  value=1.00, step=500.00, format="%.2f")
            revenue          = st.number_input("Revenue (₹)",          min_value=0.00,   value=1.00, step=500.00, format="%.2f",
                                           help="Used to auto-calculate ROI = (Revenue - Cost) / Cost")
            new_calc_roi = (revenue - acquisition_cost) / acquisition_cost if acquisition_cost > 0 else 0.0
            st.info(f"Calculated ROI: **{new_calc_roi:.4f}** ({new_calc_roi*100:.2f}%)")
            st.markdown('')

        submitted = st.form_submit_button("🔍  Predict Profit / Loss")

    if submitted:
        channel_flags = {f"Channel_Used_{ch}": (1 if ch in channels else 0) for ch in CHANNELS}

        feature_row = {
        "Impressions":              impressions,
        "Clicks":                   clicks,
        "Leads":                    leads,
        "Acquisition_Cost":         acquisition_cost,
        "New_Calc_ROI":             new_calc_roi,
        "Engagement_Score":         eng_score,
        "Conversions":              conversions,
        "Campaign_Encoded":         CAMPAIGN_MAP[campaign],
        "Campaign_Type_Encoded":    CAMPAIGN_TYPE_MAP[campaign_type],
        "Target_Audience_Encoded":  TARGET_AUDIENCE_MAP[target_audience],
        "Language_Encoded":         LANGUAGE_MAP[language],
        "Customer_Segment_Encoded": CUSTOMER_SEGMENT_MAP[customer_segment],
        "Channel_Used_Email":       channel_flags["Channel_Used_Email"],
        "Channel_Used_Facebook":    channel_flags["Channel_Used_Facebook"],
        "Channel_Used_Google":      channel_flags["Channel_Used_Google"],
        "Channel_Used_Instagram":   channel_flags["Channel_Used_Instagram"],
        "Channel_Used_WhatsApp":    channel_flags["Channel_Used_WhatsApp"],
        "Channel_Used_YouTube":     channel_flags["Channel_Used_YouTube"],
        "Duration":                 duration,
        "Day":                      campaign_date.day,
        "Month":                    campaign_date.month,
        "Year":                     campaign_date.year,
        }

        X_input  = pd.DataFrame([feature_row])[CLS_FEATURES]
        X_scaled = scaler_cls.transform(X_input)

        prediction = log_model.predict(X_scaled)[0]          # 0 = Loss, 1 = Profit
        proba      = log_model.predict_proba(X_scaled)[0]    # [P(Loss), P(Profit)]
        prob_loss, prob_profit = proba[0] * 100, proba[1] * 100

        if prediction == 1:
            st.write("Prediction Result")
            st.markdown("✅ PROFIT")
            st.markdown(f"Model confidence: {prob_profit:.1f}%")

        else:
            st.write("Prediction Result")
            st.markdown("❌ LOSS")
            st.markdown(f"Model confidence: {prob_loss:.1f}%")

        st.markdown("#### Probability Breakdown")
        profit_1, loss_1 = st.columns(2)
        with profit_1:
            st.markdown('📈 Profit Probability')
            st.progress(int(prob_profit))
            st.markdown(f"**{prob_profit:.2f}%**")
        with loss_1:
            st.markdown('📉 Loss Probability')
            st.progress(int(prob_loss))
            st.markdown(f"**{prob_loss:.2f}%**")

        st.markdown("#### Campaign Summary")
        profit_val = revenue - acquisition_cost
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue",          f"₹ {revenue:,.2f}")
        c2.metric("Acquisition Cost", f"₹ {acquisition_cost:,.2f}")
        c3.metric("Profit / Loss",    f"₹ {profit_val:,.2f}", delta=f"{profit_val:,.2f}")
        c4.metric("ROI",              f"{new_calc_roi*100:.2f}%")

        with st.expander("📋 Full Input Summary"):
            st.dataframe(
                pd.DataFrame({"Feature": list(feature_row.keys()), "Value": list(feature_row.values())}),
                use_container_width=True, hide_index=True
                         )

    st.markdown("---")
    st.caption("Logistic Regression Classification · Profit / Loss Prediction · Nykaa · Purplle · Tira")



    