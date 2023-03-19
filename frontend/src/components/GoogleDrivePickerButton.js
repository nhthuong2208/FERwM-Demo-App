import { Button } from 'antd';
import React, { useState, useEffect } from 'react';

const CLIENT_ID = '815340705842-gnqt5fbqdgv2g8m5gjoa152pdu5uibqd.apps.googleusercontent.com';
const API_KEY = 'AIzaSyAa9L0LfmcjT3-49hPeqN3IBRAzQLFfJoo';
const SCOPES = 'https://www.googleapis.com/auth/drive.file https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/userinfo.email';
const APP_ID = '815340705842';

function GoogleDrivePickerButton() {
  const [picker, setPicker] = useState(null);
  const [pickerInited, setPickerInited] = useState(false)
  const [gisInited, setGisInited] = useState(false)
  const [tokenClient, setTokenClient] = useState()
  const [accessToken, setAccessToken] = useState(null);
  let token = null

  useEffect(() => {
    const script = document.createElement('script')
    const script1 = document.createElement('script')

    script.src = "https://apis.google.com/js/api.js"
    script.defer = true
    script.async = true
    script.onload = gapiLoaded

    script1.src = "https://accounts.google.com/gsi/client"
    script1.defer = true
    script1.async = true
    script1.onload = gisLoaded

    document.body.appendChild(script)
    document.body.appendChild(script1)

    function gapiLoaded() {
      window.gapi.load('client:picker', initializePicker);
    }

    /**
     * Callback after the API client is loaded. Loads the
     * discovery doc to initialize the API.
     */
    async function initializePicker() {
      await window.gapi.client.load('https://www.googleapis.com/discovery/v1/apis/drive/v3/rest');
      setPickerInited(true)
      maybeEnableButtons();
    }

    /**
     * Callback after Google Identity Services are loaded.
     */
    function gisLoaded() {
      setTokenClient(window.google.accounts.oauth2.initTokenClient({
        client_id: CLIENT_ID,
        scope: SCOPES,
        callback: '', // defined later
      }));
      setGisInited(true)
      maybeEnableButtons();
    }
  }, []);

  /**
   * Enables user interaction after all libraries are loaded.
   */
  function maybeEnableButtons() {
    if (pickerInited && gisInited) {
      document.getElementById('authorize_button').style.visibility = 'visible';
    }
  }

  /**
   *  Sign in the user upon button click.
   */
  function handleAuthClick() {
    tokenClient.callback = async (response) => {
      if (response.error !== undefined) {
        throw (response);
      }
      setAccessToken(token => {
        token = response.access_token
        
        return token
      })

      token = response.access_token
      
  
      // document.getElementById('signout_button').style.visibility = 'visible';
      // document.getElementById('authorize_button').innerText = 'Refresh';
      await createPicker();
    };

    if (accessToken === null) {
      // Prompt the user to select a Google Account and ask for consent to share their data
      // when establishing a new session.
      tokenClient.requestAccessToken({ prompt: 'consent' });
    } else {
      // Skip display of account chooser and consent dialog for an existing session.
      tokenClient.requestAccessToken({ prompt: '' });
    }
  }

  /**
   *  Sign out the user upon button click.
   */
  function handleSignoutClick() {
    if (accessToken) {
      accessToken = null;
      window.google.accounts.oauth2.revoke(accessToken);
      document.getElementById('content').innerText = '';
      document.getElementById('authorize_button').innerText = 'Authorize';
      document.getElementById('signout_button').style.visibility = 'hidden';
    }
  }

  /**
   *  Create and render a Picker object for searching images.
   */
  async function createPicker() {
    console.log(token)
    const view = new window.google.picker.View(window.google.picker.ViewId.DOCS);
    // view.setMimeTypes('image/png,image/jpeg,image/jpg');
    const picker = new window.google.picker.PickerBuilder()
      .enableFeature(window.google.picker.Feature.NAV_HIDDEN)
      .enableFeature(window.google.picker.Feature.MULTISELECT_ENABLED)
      .setDeveloperKey(API_KEY)
      .setAppId(APP_ID)
      .setOAuthToken(token)
      .addView(view)
      .addView(new window.google.picker.DocsUploadView())
      .setCallback(pickerCallback)
      .build();
    picker.setVisible(true);
  }

  /**
   * Displays the file details of the user's selection.
   * @param {object} data - Containers the user selection from the picker
   */
  async function pickerCallback(data) {
    if (data.action === window.google.picker.Action.PICKED) {
      let text = `Picker response: \n${JSON.stringify(data, null, 2)}\n`;
      const document = data[window.google.picker.Response.DOCUMENTS][0];
      const fileId = document[window.google.picker.Document.ID];
      console.log(fileId);
      const res = await window.gapi.client.drive.files.get({
        'fileId': fileId,
        'fields': '*',
      });
      text += `Drive API response for first document: \n${JSON.stringify(res.result, null, 2)}\n`;
      window.document.getElementById('content').innerText = text;
    }
  }

  return (
    <Button type='primary' onClick={handleAuthClick}>Choose model from Google Drive</Button>
  );
}

export default GoogleDrivePickerButton;
