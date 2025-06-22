def build_html_table(summaries):
    html = """
    <div style="zoom:0.9;-ms-zoom:0.9;-webkit-transform:scale(0.9);-webkit-transform-origin:0 0;">
        <div style="background:#1a1a1a;padding:20px;margin:0;font-family:'Segoe UI',Arial,sans-serif;">
            <div style="padding-top:20px;">
                <div style="font-family:Calibri,sans-serif;font-size:24px;font-weight:bold;color:#fff;text-align:center;letter-spacing:2px;margin-bottom:20px;">
                    EXECUTIVE SUMMARY
                </div>
                <div style="height:15px;"></div>
                <div style="font-family:'Arial',Arial,sans-serif;font-size:18px;font-weight:bold;color:#fff;text-align:center;text-decoration:underline;margin-bottom:25px;">
                    <u>Overall Positive Comments across Customer Journeys</u>
                </div>
                <div style="height:20px;"></div>
            </div>
            <table cellpadding="0" cellspacing="0" style="width:100%;border-collapse:collapse;border:2px solid #fff;">
                <tr style="background:#1a1a1a;">
                    <th style="font-family:'Arial',Arial,sans-serif;font-size:14px;font-weight:bold;color:#fff;border:2px solid #fff;padding:12px 8px;text-align:left;width:15%;">
                        Journey
                    </th>
                    <th style="font-family:'Arial',Arial,sans-serif;font-size:14px;font-weight:bold;color:#fff;border:2px solid #fff;padding:12px 8px;text-align:left;width:65%;">
                        Positive Summary
                    </th>
                    <th style="font-family:'Arial',Arial,sans-serif;font-size:14px;font-weight:bold;color:#fff;border:2px solid #fff;padding:12px 8px;text-align:center;width:20%;">
                        CSAT Score
                    </th>
                </tr>
    """
    
    # Check if summaries is empty or None
    if not summaries:
        html += """
            <tr style="background:#1a1a1a;">
                <td colspan="3" style="font-family:'Arial',Arial,sans-serif;font-size:12px;font-weight:normal;color:#87CEEB;border:2px solid #fff;padding:12px 8px;text-align:center;">
                    No data available
                </td>
            </tr>
        """
    else:
        for journey, summary in summaries.items():
            # Handle different summary formats
            if isinstance(summary, str):
                # Split summary into lines and clean them
                summary_lines = [line.strip() for line in summary.split('\n') if line.strip()]
                # Remove bullet points and empty lines
                summary_lines = [line.strip('• ').strip() for line in summary_lines if line.strip('• ').strip()]
                # Skip header lines if they exist
                if len(summary_lines) > 2 and ('positive' in summary_lines[0].lower() or 'summary' in summary_lines[0].lower()):
                    summary_lines = summary_lines[2:]
            else:
                summary_lines = [str(summary)]
            
            # Ensure we have at least one line
            if not summary_lines:
                summary_lines = ["No summary available"]
            
            # Get CSAT score for this journey
            csat_score = CSAT_SCORES.get(journey, "") if 'CSAT_SCORES' in globals() else ""
            
            # Calculate rowspan for the journey cell
            rowspan = len(summary_lines)
            
            for i, line in enumerate(summary_lines):
                html += f"""
                    <tr style="background:#1a1a1a;">
                """
                
                if i == 0:
                    html += f"""
                        <td style="font-family:'Arial',Arial,sans-serif;font-size:12px;font-weight:normal;color:#87CEEB;border:2px solid #fff;padding:12px 8px;vertical-align:top;text-align:left;" rowspan="{rowspan}">
                            {journey}
                        </td>
                    """
                
                html += f"""
                        <td style="font-family:'Arial',Arial,sans-serif;font-size:12px;font-weight:normal;color:#87CEEB;border:2px solid #fff;padding:12px 8px;text-align:left;">
                            {line}
                        </td>
                """
                
                if i == 0:
                    html += f"""
                        <td style="font-family:'Arial',Arial,sans-serif;font-size:12px;font-weight:normal;color:#87CEEB;border:2px solid #fff;padding:12px 8px;text-align:center;vertical-align:top;" rowspan="{rowspan}">
                            {csat_score}
                        </td>
                    """
                
                html += """
                    </tr>
                """
    
    html += """
            </table>
        </div>
    </div>
    """
    
    return html
