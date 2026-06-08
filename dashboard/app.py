import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px  
import matplotlib.pyplot as plt

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
    "**Dashboard:**", 
     options=["Campaign_Revenue","Confusion Matrix","Revenue by Brand and Year","ROI by Campaign Type and Brand","Total ROI by Campaign Type and Brand","profit vs Loss count by Brand","correlation analysis","Regression Model Data","Residuals chart","Features Analysis","Model Comparison"]
)
    if selection == "Campaign_Revenue":
        campaign_revenue_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\revenue_brand_analysis_chart.pkl")
        st.plotly_chart(campaign_revenue_fig)

    elif selection == "Revenue by Brand and Year":
        roi_campaign_year_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\revenue_year_brand_chart.pkl")
        st.plotly_chart(roi_campaign_year_fig, use_container_width=True)

    elif selection == "ROI by Campaign Type and Brand":
        roi_campaign_type_and_campaign_analysis_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\roi_campaign_type_analysis_chart.pkl")
        st.plotly_chart(roi_campaign_type_and_campaign_analysis_fig, use_container_width=True)

    elif selection == "Total ROI by Campaign Type and Brand":
        tot_roi_campaign_type_and_campaign_analysis_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\total_roi_campaign_type_analysis_chart.pkl")
        st.plotly_chart(tot_roi_campaign_type_and_campaign_analysis_fig, use_container_width=True)


    elif selection == "profit vs Loss count by Brand":
        col1, col2 = st.columns(2)
        with col1:
            profit_loss_campaign_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\profit_loss_campaign_analysis_chart.pkl")
            st.plotly_chart(profit_loss_campaign_fig, use_container_width=True)
        with col2:
            profit_loss_pie_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\profit_loss_pie_chart.pkl")
            st.plotly_chart(profit_loss_pie_fig, use_container_width=True)


    elif selection == "correlation analysis":
        correlation_analysis_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\regres_correlation_heatmap_chart.pkl")
        st.plotly_chart(correlation_analysis_fig, use_container_width=True)


    elif selection == "Regression Model Data":
        col1, col2 = st.columns(2)
        with col1:
            reg_rf_test_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\reg_rf_test_actual_pred_chart.pkl")
            st.plotly_chart(reg_rf_test_fig , use_container_width=True)

        with col2:
            reg_rf_train_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\reg_rf_train_actual_pred_chart.pkl")
            st.plotly_chart(reg_rf_train_fig, use_container_width=True)


    elif selection == "Residuals chart":
        residual_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\residual_chart.pkl")
        st.plotly_chart(residual_fig, use_container_width=True)
    

    elif selection == "Features Analysis":
        feature_analysis_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\feature_importance_chart.pkl")
        st.plotly_chart(feature_analysis_fig, use_container_width=True)
    

    elif selection == "Model Comparison":
        col1, col2 = st.columns(2)
        with col1:
            reg_model_compare_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\reg_model_comparison_chart.pkl")
            st.plotly_chart(reg_model_compare_fig, use_container_width=True)      
        with col2:
            cls_model_compare_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\cls_model_comparison_chart.pkl")
            st.plotly_chart(cls_model_compare_fig, use_container_width=True)
    
    elif selection == "Confusion Matrix":
        col1, col2 = st.columns(2)
        with col1:
            confusion_train_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\confusion_matrix_heatmap_train_chart.pkl")
            st.plotly_chart(confusion_train_fig, use_container_width=True)      
        with col2:
            confusion_test_fig = joblib.load(r"C:\Users\jagad\Documents\my_classes\Tasks\mini_project_3-marketing_campaign_performance_prediction\models\confusion_matrix_heatmap_test_chart.pkl")
            st.plotly_chart(confusion_test_fig, use_container_width=True)

      


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
    'Impressions', 'Clicks', 'Leads', 'Acquisition_Cost',
    'Engagement_Score', 'Conversions','New_Calc_ROI', 'Campaign_Encoded', 'Campaign_Type_Encoded',
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
            duration      = st.number_input("Campaign Duration (days)", min_value=1, max_value=365, value=1)
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
        "Engagement_Score":         eng_score,
        "Conversions":              conversions,
        "New_Calc_ROI":             new_calc_roi,
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

        prediction = log_model.predict(X_scaled)[0]         # 0 = Loss, 1 = Profit
        proba      = log_model.predict_proba(X_scaled)[0]    # [P(Loss), P(Profit)]

        prob_loss = round(proba[0] * 100,2)
        prob_profit = round(proba[1] * 100,2)

        # Apply the same logic used to create Profit_Loss
        prediction = 1 if new_calc_roi > 0 else 0

        st.write("Prediction Result")

        if prediction == 1:
            st.markdown("✅ PROFIT")
            # st.markdown(f"Model confidence: {prob_profit:.2f}%")
        else:
            st.markdown("❌ LOSS")
            # st.markdown(f"Model confidence: {prob_loss:.2f}%")

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





    