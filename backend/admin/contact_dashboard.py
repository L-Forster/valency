"""
Admin dashboard for viewing and managing contact form submissions.
This is a simple FastAPI-based admin panel that can be accessed at /admin/contacts.
"""

import os
import sys
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import modules
backend_dir = Path(__file__).parent.parent
project_dir = backend_dir.parent
sys.path.append(str(project_dir))
sys.path.append(str(backend_dir))

from fastapi import APIRouter, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import sqlite3

from backend.contact_store import get_all_contacts, get_contact_by_id, update_contact_status

# Initialize templates directory
templates_dir = Path(__file__).parent / "templates"
templates_dir.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Create admin router
router = APIRouter(prefix="/admin")

# Simple static HTML admin template
ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Contact Submissions - Admin Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #4338ca;
            color: white;
            padding: 1rem;
            margin-bottom: 2rem;
        }
        h1 {
            margin: 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 2rem;
        }
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f3f4f6;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f9fafb;
        }
        .status-new {
            color: #2563eb;
            font-weight: bold;
        }
        .status-responded {
            color: #10b981;
        }
        .status-spam {
            color: #ef4444;
        }
        .button {
            display: inline-block;
            padding: 0.5rem 1rem;
            background-color: #4338ca;
            color: white;
            border-radius: 0.25rem;
            text-decoration: none;
            margin-right: 0.5rem;
        }
        .button:hover {
            background-color: #3730a3;
        }
        .button-red {
            background-color: #ef4444;
        }
        .button-red:hover {
            background-color: #dc2626;
        }
        .button-green {
            background-color: #10b981;
        }
        .button-green:hover {
            background-color: #059669;
        }
        .contact-details {
            background-color: #f9fafb;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .contact-message {
            background-color: white;
            padding: 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 0.25rem;
            white-space: pre-wrap;
        }
        .contact-meta {
            color: #6b7280;
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <header>
        <h1>Contact Submissions Admin</h1>
    </header>
    
    {% if contact %}
        <h2>Contact Details</h2>
        <div class="contact-details">
            <p><strong>ID:</strong> {{ contact.id }}</p>
            <p><strong>Company:</strong> {{ contact.company }}</p>
            <p><strong>Email:</strong> <a href="mailto:{{ contact.email }}">{{ contact.email }}</a></p>
            {% if contact.website %}
                <p><strong>Website:</strong> <a href="{{ contact.website }}" target="_blank">{{ contact.website }}</a></p>
            {% endif %}
            <p><strong>Status:</strong> <span class="status-{{ contact.status }}">{{ contact.status }}</span></p>
            <p><strong>Submitted:</strong> {{ contact.created_at }}</p>
            <p><strong>IP Address:</strong> {{ contact.ip_address or 'Not recorded' }}</p>
            <p><strong>User Agent:</strong> {{ contact.user_agent or 'Not recorded' }}</p>
            
            <h3>Message:</h3>
            <div class="contact-message">{{ contact.message }}</div>
            
            <h3>Actions:</h3>
            <form method="POST" action="/admin/contacts/{{ contact.id }}/update-status">
                <select name="status">
                    <option value="new" {% if contact.status == 'new' %}selected{% endif %}>New</option>
                    <option value="responded" {% if contact.status == 'responded' %}selected{% endif %}>Responded</option>
                    <option value="spam" {% if contact.status == 'spam' %}selected{% endif %}>Spam</option>
                </select>
                <button type="submit" class="button">Update Status</button>
            </form>
            
            <p>
                <a href="/admin/contacts" class="button">Back to List</a>
                <a href="mailto:{{ contact.email }}?subject=Re: Your Inquiry to Tone Analytics" class="button button-green">Reply via Email</a>
            </p>
        </div>
    {% else %}
        <h2>All Contacts</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Date</th>
                    <th>Company</th>
                    <th>Email</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {% for contact in contacts %}
                    <tr>
                        <td>{{ contact.id }}</td>
                        <td>{{ contact.created_at.split('T')[0] }}</td>
                        <td>{{ contact.company }}</td>
                        <td><a href="mailto:{{ contact.email }}">{{ contact.email }}</a></td>
                        <td class="status-{{ contact.status }}">{{ contact.status }}</td>
                        <td>
                            <a href="/admin/contacts/{{ contact.id }}" class="button">View</a>
                        </td>
                    </tr>
                {% endfor %}
            </tbody>
        </table>
    {% endif %}
</body>
</html>
"""

# Save the template to the templates directory
with open(os.path.join(templates_dir, "contacts.html"), "w") as f:
    f.write(ADMIN_TEMPLATE)

# Admin authentication for simplicity
# In production, use proper authentication with sessions and secure storage
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpassword")

def admin_auth(request: Request):
    """
    Simple HTTP Basic authentication for admin routes.
    In production, use a more secure authentication method.
    """
    # Implement proper authentication here
    # For now we're just checking if the user is hitting the admin endpoint
    # In production, use sessions and secure authentication
    return True

@router.get("/contacts", response_class=HTMLResponse)
async def list_contacts(request: Request, authenticated: bool = Depends(admin_auth)):
    """
    List all contact form submissions in a simple HTML admin interface.
    """
    contacts = get_all_contacts()
    
    # Render template
    return templates.TemplateResponse(
        "contacts.html",
        {"request": request, "contacts": contacts, "contact": None}
    )

@router.get("/contacts/{contact_id}", response_class=HTMLResponse)
async def view_contact(
    request: Request, 
    contact_id: int, 
    authenticated: bool = Depends(admin_auth)
):
    """
    View details of a specific contact form submission.
    """
    contact = get_contact_by_id(contact_id)
    
    if not contact:
        raise HTTPException(status_code=404, detail=f"Contact with ID {contact_id} not found")
    
    # Render template
    return templates.TemplateResponse(
        "contacts.html",
        {"request": request, "contact": contact, "contacts": []}
    )

@router.post("/contacts/{contact_id}/update-status", response_class=RedirectResponse)
async def update_status(
    request: Request,
    contact_id: int,
    status: str = Form(...),
    authenticated: bool = Depends(admin_auth)
):
    """
    Update the status of a contact.
    """
    if status not in ["new", "responded", "spam"]:
        raise HTTPException(status_code=400, detail="Invalid status value")
    
    success = update_contact_status(contact_id, status)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Contact with ID {contact_id} not found")
    
    # Redirect back to the contact detail page
    return RedirectResponse(url=f"/admin/contacts/{contact_id}", status_code=303)


# This function integrates the admin router with the main FastAPI app
def setup_admin_routes(app):
    """
    Set up admin routes in the main FastAPI app.
    """
    app.include_router(router)
    print("Admin routes set up at /admin/contacts")

if __name__ == "__main__":
    # For testing admin dashboard directly
    import uvicorn
    from fastapi import FastAPI
    
    app = FastAPI()
    setup_admin_routes(app)
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
